from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from . import register_method
from .er import ER
from ..data import ReplayBuffer
from ..loggers import BaseLogger
from ..models import ContinualModel, ContinualAngularModel, SupConLoss


@register_method("der")
class DER(ER):
    def __init__(
        self,
        model: Union[ContinualModel, ContinualAngularModel, torch.nn.Module],
        optim: torch.optim,
        buffer: ReplayBuffer,
        logger: Optional[BaseLogger] = None,
        n_replay_samples: Optional[int] = None,
        **kwargs,
    ) -> "DER":
        """_summary_

        Args:
            model (Union[ContinualModel, ContinualAngularModel, torch.nn.Module]): _description_
            optim (torch.optim): _description_
            buffer (ReplayBuffer): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            n_replay_samples (Optional[int], optional): _description_. Defaults to None.

        Returns:
            DER: _description_
        """
        super().__init__(
            model=model,
            optim=optim,
            buffer=buffer,
            logger=logger,
            n_replay_samples=n_replay_samples,
        )

        self.alpha = .5

    @property
    def name(self) -> str:
        return "der"

    def process_inc(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:
        """_summary_

        Args:
            x (torch.FloatTensor): _description_
            y (torch.FloatTensor): _description_
            t (torch.FloatTensor): _description_

        Returns:
            torch.FloatTensor: _description_
        """

        # process data
        logits = self.model(x, y, t)
        loss = self.loss(logits, y)

        return loss, logits.detach()

    def process_re(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        t: torch.FloatTensor,
        re_logits: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """_summary_

        Args:
            x (torch.FloatTensor): _description_
            y (torch.FloatTensor): _description_
            t (torch.FloatTensor): _description_
            re_logits (torch.FloatTensor): _description_

        Returns:
            torch.FloatTensor: _description_
        """

        logits = self.model(x, y, t)
        loss = self.alpha * F.mse_loss(logits, re_logits)

        return loss

    def observe(self, 
            x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, 
            not_aug_x: Optional[torch.FloatTensor] = None
        ) -> torch.FloatTensor:
        inc_loss, logits = self.process_inc(x, y, t)

        re_loss = 0
        if len(self.buffer) > 0:

            if self.n_replay_samples is None:
                self.n_replay_samples = x.size(0)

            re_data = self.buffer.sample(n_samples=self.n_replay_samples)
            re_loss = self.process_re(
                x=re_data["x"],
                y=re_data["y"],
                t=re_data["t"],
                re_logits=re_data["logits"],
            )

        loss = inc_loss + re_loss
        self.update(loss)

        self.buffer.add(batch={"x": x, "y": y, "t": t, "logits": logits})

        return loss
    
    def supcon_observe(self, 
        x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, 
        not_aug_x: torch.FloatTensor
    ) -> torch.FloatTensor:
        assert isinstance(self.loss, SupConLoss), "supcon_observe function must have SupconLoss"
        bsz = y.shape[0]

        # compute loss 
        logits = self.model(x, y, t)
        f1, f2 = torch.split(logits, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        inc_loss = self.loss(features, y)

        re_loss = 0
        if len(self.buffer) > 0:

            if self.n_replay_samples is None:
                self.n_replay_samples = x.size(0)

            re_data = self.buffer.sample(n_samples=self.n_replay_samples)
            
            
            buf_outputs = self.model(re_data["x"], re_data["y"], re_data["t"])
            loss = self.alpha * F.mse_loss(buf_outputs, re_data["logits"])

        loss = inc_loss + re_loss
        self.update(loss)

        self.buffer.add(batch={"x": not_aug_x, "y": y, "t": t, "logits": logits.detach()})

        return loss


@register_method("derpp")
class DERpp(DER):
    def __init__(
        self,
        model: Union[ContinualModel, ContinualAngularModel, torch.nn.Module],
        optim: torch.optim,
        buffer: ReplayBuffer,
        logger: Optional[BaseLogger] = None,
        n_replay_samples: Optional[int] = None,
        **kwargs,
    ) -> "DERpp":
        """_summary_

        Args:
            model (Union[ContinualModel, ContinualAngularModel, torch.nn.Module]): _description_
            optim (torch.optim): _description_
            buffer (ReplayBuffer): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            n_replay_samples (Optional[int], optional): _description_. Defaults to None.

        Returns:
            DERpp: _description_
        """
        super().__init__(
            model=model,
            optim=optim,
            buffer=buffer,
            logger=logger,
            n_replay_samples=n_replay_samples,
        )

        self.beta = 1

    @property
    def name(self) -> str:
        return "der++"

    def process_re(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        t: torch.FloatTensor,
        re_logits: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """_summary_

        Args:
            x (torch.FloatTensor): _description_
            y (torch.FloatTensor): _description_
            t (torch.FloatTensor): _description_
            re_logits (torch.FloatTensor): _description_

        Returns:
            torch.FloatTensor: _description_
        """

        logits = self.model(x, y, t)
        o1, o2 = logits.chunk(2)

        x1, x2 = x.chunk(2)
        y1, y2 = y.chunk(2)
        l1, l2 = re_logits.chunk(2)

        aa = F.mse_loss(o1, l1)
        bb = self.loss(o2, y2)

        loss = self.alpha * aa + self.beta * bb

        return loss
