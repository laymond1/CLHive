from typing import Any, List, Optional, Tuple, Union
import torch

from . import register_method, BaseMethod
from ..loggers import BaseLogger
from ..models import ContinualModel, ContinualAngularModel


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

    def observe(self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor):
        pred = self.model(x, y, t)
        loss = self.loss(pred, y)

        self.update(loss)

        return loss
