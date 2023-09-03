import argparse
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torchvision import transforms

from clhive.loggers import BaseLogger, Logger
from clhive.data import SplitCIFAR10, SplitCIFAR100, RepresentationCIFAR10
from clhive.utils.evaluators import ContinualEvaluator, ProbeEvaluator, RepresentationEvaluator
from clhive.utils.generic import seedEverything, warmup_learning_rate, TwoCropTransform
from clhive.scenarios import ClassIncremental, TaskIncremental, RepresentationIncremental
from clhive.models import ContinualModel
from clhive.methods import auto_method
from clhive import Trainer, SupConTrainer, ReplayBuffer


def parse_option():
    parser = argparse.ArgumentParser('argument for supervised contrastive method training')

    parser.add_argument('--seed', type=int, default=2023,
                        help='set seed')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--test_batch_size', type=int, default=1024,
                        help='test_batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--n_tasks', type=int, default=5,
                        help='number of tasks')
    parser.add_argument('--n_classes_per_task', type=int, default=2,
                        help='number of classes per task')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--n_LP_epochs', type=int, default=100,
                        help='number of training epochs for the linear classifier')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='20,40',
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
                        choices=['finetuning', 'er', 'der', 'lwf', 'ewc', 'crl', 'scr', 'tncr', 'vncr', 'co2l', 'vencr', 'vco2l', 'co2l_asym', 'co2l_ird'], help='choose continual learning method')
    parser.add_argument('--buffer_capacity', type=int, default=50*10, help='buffer_capacity')
    parser.add_argument('--backbone_name', type=str, default='resnet18',
                        choices=['resnet18', None], help='choose backbone_name')
    parser.add_argument('--rep_method', type=str, default='supconmlp',
                        choices=['linear', 'arcface', 'cosface', 'sphereface', 'supconmlp'], help='choose representation learning method (head_name)')
    
    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
    ## CO2L
    parser.add_argument('--current_temp', type=float, default=0.2,)
    parser.add_argument('--past_temp', type=float, default=0.01,)
    parser.add_argument('--distill_power', type=float, default=1.0,)
    # VNCR & VENCR
    parser.add_argument('--gamma', type=float, default=0.5,)

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

if opt.dataset == 'cifar10':
    TRANSFORM10 = transforms.Compose(
                    [transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                        (0.2470, 0.2435, 0.2615))])
    TRANSFORM10_ = TwoCropTransform(
                    nn.Sequential(
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                        (0.2470, 0.2435, 0.2615))))
    test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                     (0.2470, 0.2435, 0.2615))])
    train_transform = TRANSFORM10
    # Train & Test dataset and scenario
    dataset = SplitCIFAR10(root="../cl-datasets/",
                        transform=TwoCropTransform(train_transform))
    test_dataset = SplitCIFAR10(root="../cl-datasets/", 
                        train=False,  
                        transform=test_transform)
    rep_test_dataset = RepresentationCIFAR10(root="../cl-datasets/",
                        transform=test_transform,
                        data_annot="../cl-datasets/")

elif opt.dataset == 'cifar100':
    TRANSFORM100 = transforms.Compose(
                    [transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408),
                                        (0.2675, 0.2565, 0.2761))])
    test_transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408),
                                        (0.2675, 0.2565, 0.2761))])
    train_transform = TRANSFORM100
    # Train & Test dataset and scenario
    dataset = SplitCIFAR100(root="../cl-datasets/", 
                        transform=TwoCropTransform(train_transform))
    test_dataset = SplitCIFAR100(root="../cl-datasets/", 
                        train=False,  
                        transform=test_transform)
    rep_test_dataset = RepresentationCIFAR10(root="../cl-datasets/",
                        transform=test_transform,
                        data_annot="../cl-datasets/")

method_transform = transforms.Compose(
            [transforms.ToPILImage(), 
            TwoCropTransform(train_transform)])


def main(opt):

    DEFAULT_RANDOM_SEED = opt.seed
    seedEverything(DEFAULT_RANDOM_SEED)

    scenario = ClassIncremental(
        dataset=dataset, n_tasks=opt.n_tasks, batch_size=opt.batch_size, n_workers=0
    )

    # Check the number of classes per task
    assert scenario.dataset.n_classes_per_task == opt.n_classes_per_task, \
        f"{scenario.dataset.n_classes_per_task} != {opt.n_classes_per_task} Number of classes per task is not matched."

    print(f"Number of tasks: {scenario.n_tasks} | Number of classes: {scenario.n_classes}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = ContinualModel.auto_model(backbone_name=opt.backbone_name, scenario=scenario, head_name=opt.rep_method).to(device)

    buffer = ReplayBuffer(capacity=opt.buffer_capacity, device=device)
    # Replay buffer and ER agent
    agent = auto_method(
        name=opt.cl_method,
        model=model,
        optim=SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay),
        buffer=buffer,
        transform=method_transform,
    )

    # Probe Evaluator
    test_scenario = ClassIncremental(
        test_dataset, n_tasks=opt.n_tasks, batch_size=opt.test_batch_size, n_workers=0
    )
    probe_evaluator = ProbeEvaluator(method=agent, 
                                    train_scenario=scenario, 
                                    eval_scenario=test_scenario, 
                                    n_epochs=opt.n_LP_epochs,
                                    device=device)

    # Representation Evaluator
    # n_tasks arguments 필요 없음.
    test_scenario = RepresentationIncremental(
        rep_test_dataset, n_tasks=opt.n_tasks, batch_size=opt.test_batch_size, n_workers=0
    )
    rep_evaluator = RepresentationEvaluator(method=agent, eval_scenario=test_scenario, device=device)

    evaluators = [probe_evaluator, rep_evaluator]

    # Logger 
    # TODO 반복 실험을 한다면 "5_runs" directory를 추가하여 실험 저장
    opt.save_path = './save/{}/{}_seed/{}_steps_{}_{}'.format(
            opt.dataset, opt.seed, opt.n_tasks, opt.cl_method, opt.rep_method
            )
    logger = Logger(opt.n_tasks)
    logger.open_txt(opt.save_path)
    logger.write_txt(msg=opt)

    # Trainer
    trainer = SupConTrainer(
        opt=opt, method=agent, scenario=scenario, evaluator=evaluators, n_epochs=opt.n_epochs, device=device, logger=logger
    )
    trainer.fit()


if __name__ == "__main__":
    main(opt)