from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from . import register_method, BaseMethod
from .er import ER
from ..data import ReplayBuffer
from ..loggers import BaseLogger
from ..models import ContinualModel, ContinualAngularModel, SupConLoss

from torchvision import transforms
from clhive.utils.generic import seedEverything, warmup_learning_rate, TwoCropTransform


TRANSFORM10 = nn.Sequential(
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                    (0.2470, 0.2435, 0.2615)))
twocifar10 = TwoCropTransform(TRANSFORM10)

def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)

@register_method("vncr")
class VNCR(ER):
    def __init__(
        self,
        model: Union[ContinualModel, ContinualAngularModel, torch.nn.Module],
        optim: torch.optim,
        buffer: ReplayBuffer,
        transform: Optional[torch.nn.Module] = None,
        logger: Optional[BaseLogger] = None,
        n_replay_samples: Optional[int] = None,
        **kwargs,
    ) -> "VNCR":
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

    @property
    def name(self) -> str:
        return "vncr"
    
    def supcon_observe(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, not_aug_x: torch.FloatTensor
    ) -> torch.FloatTensor:
        assert isinstance(self.loss, SupConLoss), "supcon_observe function must have SupconLoss"
        real_bsz = y.size(0)
        
        if len(self.buffer) > 0:
            re_data = self.buffer.sample(n_samples=not_aug_x.size(0))
            # concat data
            x, y, t = torch.cat([not_aug_x, re_data["x"]], dim=0), torch.cat([y, re_data["y"]], dim=0), torch.cat([t, re_data["t"]], dim=0)
            
            # virtual(mixup) negative samples
            n_not_aug_x = partial_mixup(input=x, gamma=0.5, indices=torch.randperm(x.size(0)))
            # create label and task id
            n_y = torch.ones(y.size(0), dtype=torch.long).fill_((t[0]+1)*2).to(y.device)
            n_t = torch.ones(y.size(0), dtype=torch.long).fill_(t[0]).to(t.device)
            # merge with all samples
            x = torch.cat([x, n_not_aug_x], dim=0)
        else:
            # virtual(mixup) negative samples
            n_not_aug_x = partial_mixup(input=not_aug_x, gamma=0.5, indices=torch.randperm(not_aug_x.size(0)))
            # create label and task id
            n_y = torch.ones(not_aug_x.size(0), dtype=torch.long).fill_((t[0]+1)*2).to(y.device)
            n_t = torch.ones(not_aug_x.size(0), dtype=torch.long).fill_(t[0]).to(t.device)
            # merge with all samples
            x = torch.cat([not_aug_x, n_not_aug_x], dim=0)
        
        # merge with all samples
        x = torch.cat(twocifar10(x), dim=0)            
        y, t = torch.cat([y, n_y], dim=0), torch.cat([t, n_t], dim=0)

        bsz = y.shape[0]
        # compute loss
        features = self.model(x)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.loss(features, y)
        
        self.update(loss)
        
        self.buffer.add(batch={"x": not_aug_x, "y": y[:real_bsz], "t": t[:real_bsz]})

        return loss

