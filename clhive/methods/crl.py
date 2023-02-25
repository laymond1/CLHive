from argparse import Namespace
from typing import Any, List, Optional, Tuple, Union, ClassVar
import copy
import torch

from . import register_method, BaseMethod
from ..loggers import BaseLogger
from ..models import ContinualModel, ContinualAngularModel, SupConLoss


@register_method("crl")
class CRL(BaseMethod):
    def __init__(
        self,
        model: Union[ContinualModel, ContinualAngularModel, torch.nn.Module],
        optim: torch.optim,
        logger: Optional[BaseLogger] = None,
        args: Optional[Namespace] = None,
        **kwargs,
    ) -> "CRL":
        """_summary_
        reference: <"Continual representation learning for biometric identification.">
            in IEEE/CVF Winter Conference on Applications of Computer Vision. 2021.
        
        Args:
            model (Union[ContinualModel, ContinualAngularModel, torch.nn.Module]): _description_
            optim (torch.optim): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.

        Returns:
            CRL: _description_
        """
        super().__init__(model, optim, logger)

        self.args = args #TODO this will be included in BaseMethod
        self.loss = torch.nn.CrossEntropyLoss()
        if args is not None:
            self.k = args.k
            self.temper = args.temper
            self.beta = args.beta
            self.lambda_0 = args.lambda_0
            self.prev_model = args.prev_model
        else:
            self.k = 10
            self.temper = 2
            self.beta = 10*1-3
            self.lambda_0 = 1
            self.prev_model = None

    @property
    def name(self) -> str:
        return "crl"

    def record_state(self) -> None:
        self.prev_model = copy.deepcopy(self.model)

    def _distillation_loss(
        self, 
        current_out: torch.FloatTensor, 
        prev_out: torch.FloatTensor, 
        k: int, 
        temper: int, 
        beta: float
    ) -> torch.FloatTensor:
        """
        Compute distillation loss between output(b) of the current model and
        output(a) of the previous model at top-K
        """
        _, a_indices = prev_out.sort(descending=True)
        s_set = a_indices[:, :k]

        log_q = torch.log_softmax(current_out.gather(dim=1, index=s_set)/temper, dim=1) # input (new_model)
        log_p = torch.log_softmax(prev_out.gather(dim=1, index=s_set)/temper, dim=1) # target (old_model)
        kl_loss = torch.nn.functional.kl_div(log_q, log_p, reduction='none', log_target=True)

        delta = -beta * torch.softmax(prev_out.gather(dim=1, index=s_set)/temper, dim=1) * log_p
        loss = kl_loss.sum(dim=1) - delta.sum(dim=1)

        zero = torch.zeros(1, device=loss.device)
        loss = torch.max(zero, loss).mean()

        return loss

    def crl_loss(
        self,
        features: torch.FloatTensor,
        data: torch.FloatTensor,
        y: torch.Tensor,
        current_model: torch.nn.Module,
        current_task: torch.FloatTensor,
        ) -> torch.FloatTensor:
        if self.prev_model is None:
            return 0.0

        predictions_old_tasks_old_model = dict()
        predictions_old_tasks_new_model = dict()
        for task_id in range(current_task[0]):
            with torch.inference_mode():
                predictions_old_tasks_old_model[task_id] = self.prev_model(
                    data, y, t=torch.full_like(current_task, fill_value=task_id)
                )
            predictions_old_tasks_new_model[task_id] = current_model.forward_head(
                features, y, t=torch.full_like(current_task, fill_value=task_id)
            )

        dist_loss = 0
        for task_id in predictions_old_tasks_old_model.keys():
            dist_loss += self._distillation_loss(
                current_out=predictions_old_tasks_new_model[task_id],
                prev_out=predictions_old_tasks_old_model[task_id].clone(),
                k=self.k,
                temper=self.temper,
                beta=self.beta
            )

        return dist_loss
    
    def crl_supconloss(
        self,
        features: torch.FloatTensor,
        data: torch.FloatTensor,
        y: torch.Tensor,
        current_model: torch.nn.Module,
        current_task: torch.FloatTensor,
        ) -> torch.FloatTensor:
        if self.prev_model is None:
            return 0.0

        predictions_old_tasks_old_model = dict()
        predictions_old_tasks_new_model = dict()
        for task_id in range(current_task[0]):
            with torch.inference_mode():
                predictions_old_tasks_old_model[task_id] = self.prev_model(
                    data, y, t=torch.full_like(current_task, fill_value=task_id)
                )
            predictions_old_tasks_new_model[task_id] = features

        dist_loss = 0
        for task_id in predictions_old_tasks_old_model.keys():
            dist_loss += self._distillation_loss(
                current_out=predictions_old_tasks_new_model[task_id],
                prev_out=predictions_old_tasks_old_model[task_id].clone(),
                k=self.k,
                temper=self.temper,
                beta=self.beta
            )

        return dist_loss

    def observe(self, 
            x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, 
            not_aug_x: Optional[torch.FloatTensor] = None
        ) -> torch.FloatTensor:

        features = self.model.forward_backbone(x)
        # 1. new loss
        pred = self.model.forward_head(features, y, t)
        new_loss = self.loss(pred, y)
        # 2. old loss
        crl_loss = self.crl_loss(
            features=features, data=x, y=y, current_model=self.model, current_task=t
        )
        # Total loss
        loss = new_loss + self.lambda_0 * crl_loss
        self.update(loss)

        return loss
    

    def supcon_observe(self, 
            x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, 
            not_aug_x: Optional[torch.FloatTensor] = None
        ) -> torch.FloatTensor:
        assert isinstance(self.loss, SupConLoss), "supcon_observe function must have SupconLoss"
        bsz = y.shape[0]

        # compute current model supcon loss
        new_features = self.model(x) # [512, 128]
        f1, f2 = torch.split(new_features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        new_loss = self.loss(features, y)

        crl_loss = self.crl_supconloss(
            features=new_features, data=x, y=y, current_model=self.model, current_task=t
        )

        # Total loss
        loss = new_loss + self.lambda_0 * crl_loss
        self.update(loss)

        return loss

    def on_task_end(self) -> None:
        super().on_task_end()
        self.record_state()

        