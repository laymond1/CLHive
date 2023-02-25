from typing import Any, Callable, Dict, List, Optional
from rich.console import Console
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import numpy as np 
import os 
import pandas as pd
import random
import torch


DEFAULT_RANDOM_SEED = 2023

def seedEverything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(config: Dict[str, Any]):
    assert hasattr(
        torch.optim, config.name
    ), f"{config.name} is not a registered optimizer in torch.optim"
    optim = getattr(torch.optim, config.name)(**config)
    return optim


def spinner_animation(message: str, spinner_type: Optional[str] = "dots"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            console = Console()
            with console.status(message, spinner=spinner_type):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Taken from AML codebase


def load_best_args(
    args,
    target="acc",
    avg_over="run",
    keep=["method", "use_augs", "task_free", "dataset", "mem_size", "mir_head_only"],
):
    # load the dataframe with the hparam runs
    df = pd.read_csv("sweeps/hp_result.csv")

    # subselect the appropriate runs
    for key in keep:
        df = df[df[key] == getattr(args, key)]

    # which arg to overwrite ?
    unique = df.nunique()
    arg_list = list(unique[unique > 1].index)
    arg_list.remove(avg_over)
    arg_list.remove(target)

    # find the best run
    acc_per_cfg = df.groupby(arg_list)[target].agg(["mean", "std"])
    acc_per_cfg = acc_per_cfg.rename(
        columns={"mean": f"{target}_mean", "std": f"{target}_std"}
    )
    arg_values = acc_per_cfg[f"{target}_mean"].idxmax()

    if not isinstance(arg_values, Iterable):
        arg_values = [arg_values]

    print("overwriting args")
    for (k, v) in zip(arg_list, arg_values):
        print(f"{k} from {getattr(args, k)} to {v}")
        setattr(args, k, v)


def sho_(x, nrow=8):
    x = x * 0.5 + 0.5
    from torchvision.utils import save_image
    from PIL import Image

    if x.ndim == 5:
        nrow = x.size(1)
        x = x.reshape(-1, *x.shape[2:])

    save_image(x, "tmp.png", nrow=nrow)
    Image.open("tmp.png").show()


def save_(x, name="tmp.png"):
    x = x * 0.5 + 0.5
    from torchvision.utils import save_image
    from PIL import Image

    if x.ndim == 5:
        nrow = x.size(1)
        x = x.reshape(-1, *x.shape[2:])

    save_image(x, name)


# --- metrics utils
def compute_metrics(results):
    """
    Given accuracy results list(np.array), compute all metrics performances such as 
    [Average Accuracy, Average, Forgetting, Positive Backward Transfer, Forward Transfer]
    """ 
    n_tasks = results.shape[0]
    
    # compute average accuracy
    return NotImplementedError



def backward_transfer(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = []
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i])

    return np.mean(li)


def forgetting(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)


# --- Representation utils
def get_metrics(labels, scores, FPRs):
    # eer and auc
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    roc_curve = interp1d(fpr, tpr)
    EER = 100. * brentq(lambda x : 1. - x - roc_curve(x), 0., 1.)
    AUC = 100. * metrics.auc(fpr, tpr)

    # get acc
    tnr = 1. - fpr
    # pos_num = labels.count(1)
    pos_num = len(labels.nonzero()[0])
    neg_num = len(labels) - pos_num
    ACC = 100. * max(tpr * pos_num + tnr * neg_num) / len(labels)

    # TPR @ FPR
    if isinstance(FPRs, list):
        TPRs = [
            ('TPR@FPR={}'.format(FPR), 100. * roc_curve(float(FPR)))
            for FPR in FPRs
        ]
    else:
        TPRs = []

    return [('ACC', ACC), ('EER', EER), ('AUC', AUC)] + TPRs


# --- Supcon utils
    """ For SupCon Loss """
    
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# --- MIR utils
""" For MIR """


def overwrite_grad(pp, new_grad, grad_dims):
    """
    This is used to overwrite the gradients with a new gradient
    vector, whenever violations occur.
    pp: parameters
    newgrad: corrected gradient
    grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        param.grad = torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[: cnt + 1])
        this_grad = new_grad[beg:en].contiguous().view(param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1


def get_grad_vector(pp, grad_dims):
    """
    gather the gradients in one vector
    """
    grads = torch.zeros(size=(sum(grad_dims),), device=pp[0].device)

    cnt = 0
    for param in pp:
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            grads[beg:en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


def get_future_step_parameters(this_net, grad_vector, grad_dims, lr=1):
    """
    computes theta - delta theta
    :param this_net:
    :param grad_vector:
    :return:
    """
    new_net = copy.deepcopy(this_net)
    overwrite_grad(new_net.parameters, grad_vector, grad_dims)
    with torch.no_grad():
        for param in new_net.parameters():
            if param.grad is not None:
                param.data = param.data - lr * param.grad.data
    return new_net


def get_grad_dims(self):
    self.grad_dims = []
    for param in self.net.parameters():
        self.grad_dims.append(param.data.numel())


# Taken from
# https://github.com/aimagelab/mammoth/blob/cb9a36d788d6ad051c9eee0da358b25421d909f5/models/gem.py#L34
def store_grad(params, grads, grad_dims):
    """
    This stores parameter gradients of past tasks.
    pp: parameters
    grads: gradients
    grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[: count + 1])
            grads[begin:end].copy_(param.grad.data.view(-1))
        count += 1


# Taken from
# https://github.com/aimagelab/mammoth/blob/cb9a36d788d6ad051c9eee0da358b25421d909f5/models/agem.py#L21
def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger
