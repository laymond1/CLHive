from typing import Any, List, Optional, Tuple, Union
import torch

from . import register_method, BaseMethod
from ..loggers import BaseLogger
from ..models import ContinualModel, ContinualAngularModel, SupConLoss


@register_method("finetuning")
class FineTuning(BaseMethod):
    def __init__(
        self,
        model: Union[ContinualModel, ContinualAngularModel, torch.nn.Module],
        optim: torch.optim,
        logger: Optional[BaseLogger] = None,
        **kwargs,
    ) -> "FineTuning":
        """_summary_

        Args:
            model (Union[ContinualModel, ContinualAngularModel, torch.nn.Module]): _description_
            optim (torch.optim): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.

        Returns:
            FineTuning: _description_
        """    
        super().__init__(model, optim, logger)

        self.loss = torch.nn.CrossEntropyLoss()

    @property
    def name(self):
        return "finetuning"

    def observe(self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, not_aug_x: torch.FloatTensor):
        pred = self.model(x, y, t)
        loss = self.loss(pred, y)

        self.update(loss)

        return loss

    def supcon_observe(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, not_aug_x: torch.FloatTensor
    ) -> torch.FloatTensor:
        assert isinstance(self.loss, SupConLoss), "supcon_observe function must have SupconLoss"
        bsz = y.shape[0]

        # compute loss
        features = self.model(x)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.loss(features, y)

        self.update(loss)

        return loss
