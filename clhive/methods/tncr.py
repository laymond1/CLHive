from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from . import register_method, BaseMethod
from .er import ER
from ..data import ReplayBuffer
from ..loggers import BaseLogger
from ..models import ContinualModel, ContinualAngularModel, SupConLoss

from torchvision import transforms
from clhive.data import SplitCIFAR10, SplitCIFAR100, RepresentationCIFAR10
from clhive.scenarios import ClassIncremental, TaskIncremental, RepresentationIncremental
from clhive.utils.generic import seedEverything, warmup_learning_rate, TwoCropTransform


TRANSFORM10 = nn.Sequential(
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                    (0.2470, 0.2435, 0.2615)))
twocifar10 = TwoCropTransform(TRANSFORM10)

def load_negative_data():
    TRANSFORM100 = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                    (0.2675, 0.2565, 0.2761))])
    train_transform = TRANSFORM100
    # Train negative dataset and scenario
    true_negative_dataset = SplitCIFAR100(root="../cl-datasets/", 
                        transform=train_transform)
    true_negative_samples = ClassIncremental(dataset=true_negative_dataset, n_tasks=100, batch_size=128)

    print(
    f"Number of tasks: {true_negative_samples.n_tasks} | Number of classes: {true_negative_samples.n_classes}"
    )

    return next(iter(true_negative_samples))

@register_method("tncr")
class TNCR(ER):
    def __init__(
        self,
        model: Union[ContinualModel, ContinualAngularModel, torch.nn.Module],
        optim: torch.optim,
        buffer: ReplayBuffer,
        transform: Optional[torch.nn.Module] = None,
        logger: Optional[BaseLogger] = None,
        n_replay_samples: Optional[int] = None,
        **kwargs,
    ) -> "TNCR":
        """_summary_

        Args:
            model (Union[ContinualModel, ContinualAngularModel, torch.nn.Module]): _description_
            optim (torch.optim): _description_
            buffer (ReplayBuffer): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            n_replay_samples (Optional[int], optional): _description_. Defaults to None.

        Returns:
            : _description_
        """
        super().__init__(
            model=model,
            optim=optim,
            buffer=buffer,
            transform=transform,
            logger=logger,
            n_replay_samples=n_replay_samples,
        )
        
        # Set true negative samples from CIFAR100
        self.negative_buffer = ReplayBuffer(capacity=500, device=buffer.device)
        negative_loader = load_negative_data()
        for x, y, t, not_aug_x in negative_loader:
            y, t, not_aug_x = y.to(buffer.device), t.to(buffer.device), not_aug_x.to(buffer.device)
            self.negative_buffer.add(batch={"x": not_aug_x, "y": y, "t": t})

    @property
    def name(self) -> str:
        return "tncr"
    
    def supcon_observe(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, not_aug_x: torch.FloatTensor
    ) -> torch.FloatTensor:
        assert isinstance(self.loss, SupConLoss), "supcon_observe function must have SupconLoss"
        real_bsz = y.size(0)
        
        # negative samples
        negative_re_data = self.negative_buffer.sample(n_samples=not_aug_x.size(0))
        n_not_aug_x, n_y, n_t  = negative_re_data["x"], negative_re_data["y"], negative_re_data["t"]
        # adjust label and task id
        n_y = n_y + (t[0]+1)*2
        n_t = torch.ones_like(n_t).fill_(t[0])
        
        # merge with negative samples
        x = torch.cat([not_aug_x, n_not_aug_x], dim=0)
        y, t = torch.cat([y, n_y], dim=0), torch.cat([t, n_t], dim=0)
        
        if len(self.buffer) > 0:
            re_data = self.buffer.sample(n_samples=not_aug_x.size(0))
            # concat data
            x, y, t = torch.cat([x, re_data["x"]], dim=0), torch.cat([y, re_data["y"]], dim=0), torch.cat([t, re_data["t"]], dim=0)

        # augment data
        x = torch.cat(twocifar10(x), dim=0)            

        bsz = y.shape[0]
        # compute loss
        features = self.model(x)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.loss(features, y)
        
        self.update(loss)
        
        self.buffer.add(batch={"x": not_aug_x, "y": y[:real_bsz], "t": t[:real_bsz]})

        return loss
