import argparse
import torch
from torch.optim import SGD, AdamW
from torchvision import transforms

from clhive.loggers import BaseLogger, Logger
from clhive.data import SplitCIFAR10, SplitCIFAR100, RepresentationCIFAR10, CASIAWebDataset, LFWPairDataset
from clhive.data import CASIAWebDataset, LFWPairDataset, CALFWPairDataset, CPLFWPairDataset, AGEDB30PairDataset
from clhive.utils.evaluators import ContinualEvaluator, ProbeEvaluator, RepresentationEvaluator
from clhive.utils.generic import seedEverything, warmup_learning_rate, TwoCropTransform
from clhive.scenarios import ClassIncremental, TaskIncremental, RepresentationIncremental
from clhive.models import ContinualModel, ContinualAngularModel
from clhive.methods import auto_method
from clhive import Trainer, SupConTrainer, ReplayBuffer


def parse_option():
    parser = argparse.ArgumentParser('argument for base and angualr method training')

    parser.add_argument('--seed', type=int, default=2023,
                        help='set seed')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--test_batch_size', type=int, default=512,
                        help='test_batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--n_tasks', type=int, default=5,
                        help='number of tasks')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='4,8',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    
    # dataset
    parser.add_argument('--dataset', type=str, default='casiaweb',
                        choices=['cifar10', 'cifar100', 'casiaweb', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--cl_method', type=str, default='finetuning',
                        choices=['finetuning', 'er', 'der', 'lwf', 'ewc', 'crl'], help='choose continual learning method')
    parser.add_argument('--buffer_capacity', type=int, default=50*10, help='buffer_capacity')
    parser.add_argument('--backbone_name', type=str, default='iresnet50',
                        choices=['resnet18', 'resnet50', 'iresnet50', None], help='choose backbone_name')
    parser.add_argument('--rep_method', type=str, default='arcface',
                        choices=['linear', 'arcface', 'cosface', 'sphereface', 'supconmlp'], help='choose representation learning method (head_name)')
    
    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
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

if opt.dataset == 'casiaweb':
    CASIAWEBFACE = transforms.Compose(
                    [transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))])
    train_transform = CASIAWEBFACE
    # Train & Test dataset and scenario
    dataset = CASIAWebDataset(root="../cl-datasets/CASIA-aligned/", 
                        transform=train_transform)
    lfw_test_dataset = LFWPairDataset(root="../cl-datasets/",
                        transform=None,
                        data_annot="../cl-datasets/")
    calfw_test_dataset = CALFWPairDataset(root="../cl-datasets/",
                        transform=None,
                        data_annot="../cl-datasets/")
    cplfw_test_dataset = CPLFWPairDataset(root="../cl-datasets/",
                        transform=None,
                        data_annot="../cl-datasets/")
    agedb_30_test_dataset = AGEDB30PairDataset(root="../cl-datasets/",
                        transform=None,
                        data_annot="../cl-datasets/")

else:
    raise ValueError()


def main(opt):

    DEFAULT_RANDOM_SEED = opt.seed
    seedEverything(DEFAULT_RANDOM_SEED)

    scenario = ClassIncremental(
        dataset=dataset, n_tasks=opt.n_tasks, batch_size=opt.batch_size, n_workers=0
    )

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
    )

    # Representation Evaluator: n_tasks arguments 필요 없음.
    # LFW Test Dataset
    lfw_test_scenario = RepresentationIncremental(
        lfw_test_dataset, n_tasks=opt.n_tasks, batch_size=opt.test_batch_size, n_workers=0
    )
    lfw_rep_evaluator = RepresentationEvaluator(method=agent, eval_scenario=lfw_test_scenario, device=device)
    # CALFW Test Dataset
    calfw_test_scenario = RepresentationIncremental(
        calfw_test_dataset, n_tasks=opt.n_tasks, batch_size=opt.test_batch_size, n_workers=0
    )
    calfw_rep_evaluator = RepresentationEvaluator(method=agent, eval_scenario=calfw_test_scenario, device=device)
    # CPLFW Test Dataset
    cplfw_test_scenario = RepresentationIncremental(
        cplfw_test_dataset, n_tasks=opt.n_tasks, batch_size=opt.test_batch_size, n_workers=0
    )
    cplfw_rep_evaluator = RepresentationEvaluator(method=agent, eval_scenario=cplfw_test_scenario, device=device)
    # AGEDB30 Test Dataset
    agedb_30_test_scenario = RepresentationIncremental(
        agedb_30_test_dataset, n_tasks=opt.n_tasks, batch_size=opt.test_batch_size, n_workers=0
    )
    agedb_30_rep_evaluator = RepresentationEvaluator(method=agent, eval_scenario=agedb_30_test_scenario, device=device)

    evaluators = [lfw_rep_evaluator, calfw_rep_evaluator, cplfw_rep_evaluator, agedb_30_rep_evaluator]

    # Logger 
    # TODO 반복 실험을 한다면 "5_runs" directory를 추가하여 실험 저장
    opt.save_path = './save/{}/{}_seed/{}_steps_{}_{}'.format(
            opt.dataset, opt.seed, opt.n_tasks, opt.cl_method, opt.rep_method
            )
    logger = Logger(opt.n_tasks)
    logger.open_txt(opt.save_path)
    logger.write_txt(msg=opt)

    # Trainer
    trainer = Trainer(
        opt=opt, method=agent, scenario=scenario, evaluator=evaluators, n_epochs=opt.n_epochs, device=device, logger=logger
    )
    trainer.fit()


if __name__ == "__main__":
    main(opt)