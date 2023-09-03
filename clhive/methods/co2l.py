from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from . import register_method, BaseMethod
from .er import ER
from ..data import ReplayBuffer
from ..loggers import BaseLogger
from ..models import ContinualModel, ContinualAngularModel, SupConLoss, AsymSupConLoss

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

@register_method("co2l")
class CO2L(ER):
    def __init__(
        self,
        model: Union[ContinualModel, ContinualAngularModel, torch.nn.Module],
        optim: torch.optim,
        buffer: ReplayBuffer,
        transform: Optional[torch.nn.Module] = None,
        logger: Optional[BaseLogger] = None,
        n_replay_samples: Optional[int] = None,
        **kwargs,
    ) -> "CO2L":
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
        self.model2 = None
        
    @property
    def name(self) -> str:
        return "co2l"
    
    def supcon_observe(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, not_aug_x: torch.FloatTensor
    ) -> torch.FloatTensor:
        assert isinstance(self.loss, AsymSupConLoss), "supcon_observe function must have AsymSupConLoss"
        real_bsz = y.size(0)
        
        if len(self.buffer) > 0:
            re_data = self.buffer.sample(n_samples=not_aug_x.size(0))
            # concat data
            x, y, t = torch.cat([not_aug_x, re_data["x"]], dim=0), torch.cat([y, re_data["y"]], dim=0), torch.cat([t, re_data["t"]], dim=0)
            # augment data
            x = torch.cat(twocifar10(x), dim=0)            

        bsz = y.shape[0]
        
        with torch.no_grad():
            prev_task_mask = y < self.current_task * self.opt.n_classes_per_task
            prev_task_mask = prev_task_mask.repeat(2)
        
        # compute loss
        features = self.model(x)
        
        # IRD (current)
        if self.current_task > 0:
            features1_prev_task = features

            features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), self.opt.current_temp)
            logits_mask = torch.scatter(
                torch.ones_like(features1_sim),
                1,
                torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                0
            )
            logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
            features1_sim = features1_sim - logits_max1.detach()
            row_size = features1_sim.size(0)
            logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
        
        # Asym SupCon Loss
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # loss = self.loss(features, y, target_labels=list(range((t[0]+1)*2)))
        loss = self.loss(features, y, target_labels=list(range(self.current_task * self.opt.n_classes_per_task, (self.current_task+1) * self.opt.n_classes_per_task)))
        
        # IRD (past)
        if self.current_task > 0:
            with torch.no_grad():
                features2_prev_task = self.model2(x)

                features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), self.opt.past_temp)
                logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                features2_sim = features2_sim - logits_max2.detach()
                logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)


            loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
            loss += self.opt.distill_power * loss_distill
            
        # update loss                                                                                                                    
        self.update(loss)
        
        self.buffer.add(batch={"x": not_aug_x, "y": y[:real_bsz], "t": t[:real_bsz]})

        return loss
