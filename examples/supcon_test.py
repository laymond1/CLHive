import argparse
import torch
from torch.optim import SGD, AdamW
from torchvision import transforms

from clhive.data import SplitCIFAR10, SplitCIFAR100, RepresentationCIFAR10
from clhive.utils.evaluators import ContinualEvaluator, ProbeEvaluator, RepresentationEvaluator
from clhive.utils.generic import seedEverything, warmup_learning_rate
from clhive.scenarios import ClassIncremental, TaskIncremental, RepresentationIncremental
from clhive.models import ContinualModel
from clhive.methods import auto_method
from clhive import Trainer, SupConTrainer, ReplayBuffer

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
normalize = transforms.Normalize(mean=mean, std=std)
train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


DEFAULT_RANDOM_SEED = 2023
seedEverything(DEFAULT_RANDOM_SEED)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--n_tasks', type=int, default=5,
                        help='number of tasks')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--cl_method', type=str, default='finetuning',
                        choices=['finetuning', 'er', 'der', 'lwf', 'ewc', 'crl'], help='choose continual learning method')
    parser.add_argument('--buffer_capacity', type=int, default=50*10, help='buffer_capacity')
    parser.add_argument('--backbone_name', type=str, default='resnet18',
                        choices=['resnet18', None], help='choose backbone_name')
    parser.add_argument('--rep_method', type=str, default='supconmlp',
                        choices=['linear', 'arcface', 'cosface', 'sphereface', 'supconmlp'], help='choose representation learning method (head_name)')
    
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    return opt


opt = parse_option()

TRANSFORM10 = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                     (0.2470, 0.2435, 0.2615))])
test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                     (0.2470, 0.2435, 0.2615))])
method_transform = transforms.Compose(
            [transforms.ToPILImage(), 
            TwoCropTransform(TRANSFORM10)])


def main(opt):

    # HParams
    batch_size = opt.batch_size
    n_tasks = opt.n_tasks
    n_epochs = opt.n_epochs
    buffer_capacity = opt.buffer_capacity
    backbone_name = opt.backbone_name
    head_name = opt.rep_method
    cl_method = opt.cl_method

    # Train dataset and scenario
    dataset = SplitCIFAR10(root="../cl-datasets/", 
                        transform=TwoCropTransform(TRANSFORM10))

    scenario = ClassIncremental(
        dataset=dataset, n_tasks=n_tasks, batch_size=batch_size, n_workers=0
    )

    print(f"Number of tasks: {scenario.n_tasks} | Number of classes: {scenario.n_classes}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = ContinualModel.auto_model(backbone_name=backbone_name, scenario=scenario, head_name=head_name).to(device)

    buffer = ReplayBuffer(capacity=buffer_capacity, device=device)
    # Replay buffer and ER agent
    agent = auto_method(
        name=cl_method,
        model=model,
        optim=SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4),
        buffer=buffer,
        transform=method_transform,
    )

    # Base Evaluator
    test_dataset = SplitCIFAR10(root="../cl-datasets/", 
                        transform=test_transform)
    test_scenario = ClassIncremental(
        test_dataset, n_tasks=n_tasks, batch_size=batch_size, n_workers=0
    )
    base_evaluator = ContinualEvaluator(method=agent, eval_scenario=test_scenario, device=device)

    # Probe Evaluator
    test_scenario = ClassIncremental(
        test_dataset, n_tasks=n_tasks, batch_size=batch_size, n_workers=0
    )
    probe_evaluator = ProbeEvaluator(method=agent, 
                                    train_scenario=scenario, 
                                    eval_scenario=test_scenario, 
                                    n_epochs=n_epochs,
                                    device=device)

    # Representation Evaluator
    test_dataset = RepresentationCIFAR10(root="../cl-datasets/",
                                        transform=test_transform,
                                        data_annot="../cl-datasets/")
    # n_tasks arguments 필요 없음.
    test_scenario = RepresentationIncremental(
        test_dataset, n_tasks=n_tasks, batch_size=batch_size, n_workers=0
    )
    rep_evaluator = RepresentationEvaluator(method=agent, eval_scenario=test_scenario, device=device)

    # evaluators = [base_evaluator, probe_evaluator, rep_evaluator]
    evaluators = [probe_evaluator, rep_evaluator]

    # Trainer
    trainer = SupConTrainer(
        opt=opt, method=agent, scenario=scenario, evaluator=evaluators, n_epochs=n_epochs, device=device
    )
    trainer.fit()


if __name__ == "__main__":
    main(opt)