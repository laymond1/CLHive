from typing import Any, List, Optional, Tuple, Union
import torch

from . import register_method, BaseMethod
from ..data import ReplayBuffer
from ..loggers import BaseLogger
from ..models import ContinualModel, ContinualAngularModel, SupConLoss


@register_method("er")
class ER(BaseMethod):
    def __init__(
        self,
        model: Union[ContinualModel, ContinualAngularModel, torch.nn.Module],
        optim: torch.optim,
        buffer: ReplayBuffer,
        logger: Optional[BaseLogger] = None,
        n_replay_samples: Optional[int] = None,
        **kwargs,
    ) -> "ER":
        """_summary_

        Args:
            model (Union[ContinualModel, ContinualAngularModel, torch.nn.Module]): _description_
            optim (torch.optim): _description_
            buffer (ReplayBuffer): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            n_replay_samples (Optional[int], optional): _description_. Defaults to None.

        Returns:
            ER: _description_
        """
        super().__init__(model, optim, logger)

        self.buffer = buffer
        self.n_replay_samples = n_replay_samples
        self.loss = torch.nn.CrossEntropyLoss()

    @property
    def name(self) -> str:
        return "er"

    def process_inc(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:
        pred = self.model(x, y, t)
        loss = self.loss(pred, y)

        return loss

    def process_re(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:
        pred = self.model(x, y, t)
        loss = self.loss(pred, y)

        return loss

    def observe(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, not_aug_x: torch.FloatTensor
    ) -> torch.FloatTensor:
        
        if len(self.buffer) > 0:

            if self.n_replay_samples is None:
                self.n_replay_samples = x.size(0)

            re_data = self.buffer.sample(n_samples=self.n_replay_samples)
            # concat data
            x, y, t = torch.cat([x, re_data["x"]], dim=0), torch.cat([y, re_data["y"]], dim=0), torch.cat([t, re_data["t"]], dim=0)

        # compute loss
        pred = self.model(x, y, t)
        loss = self.loss(pred, y)

        self.update(loss)

        self.buffer.add(batch={"x": x, "y": y, "t": t})

        return loss

    def supcon_observe(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, not_aug_x: torch.FloatTensor
    ) -> torch.FloatTensor:
        assert isinstance(self.loss, SupConLoss), "supcon_observe function must have SupconLoss"

        if len(self.buffer) > 0:

            if self.n_replay_samples is None:
                self.n_replay_samples = not_aug_x.size(0)

            re_data = self.buffer.sample(n_samples=self.n_replay_samples)
            # concat data
            x, y, t = torch.cat([not_aug_x, re_data["x"]], dim=0), torch.cat([y, re_data["y"]], dim=0), torch.cat([t, re_data["t"]], dim=0)
            bsz = y.shape[0]
            
            # transform images one by one in buffer
            x_0s, x_1s = [], []
            for ee in x: 
                x_0, x_1 = self.transform(ee.cpu())
                x_0s.append(x_0)
                x_1s.append(x_1)
            x = torch.cat([torch.stack(x_0s), torch.stack(x_1s)], dim=0).to(y.device)

        bsz = y.shape[0]
        # compute loss
        features = self.model(x)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.loss(features, y)

        self.update(loss)

        self.buffer.add(batch={"x": not_aug_x, "y": y, "t": t})

        return loss
