from typing import Any, List, Optional, Tuple, Union
import copy
import torch

from . import register_method, BaseMethod
from ..loggers import BaseLogger
from ..models import ContinualModel, ContinualAngularModel, SupConLoss


@register_method("lwf")
class LwF(BaseMethod):
    def __init__(
        self,
        model: Union[ContinualModel, ContinualAngularModel, torch.nn.Module],
        optim: torch.optim,
        logger: Optional[BaseLogger] = None,
        **kwargs,
    ) -> "LwF":
        """_summary_

        Args:
            model (Union[ContinualModel, ContinualAngularModel, torch.nn.Module]): _description_
            optim (torch.optim): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.

        Returns:
            LwF: _description_
        """    
        super().__init__(model, optim, logger)

        self.loss = torch.nn.CrossEntropyLoss()

        self.temp = 2
        self.lambda_0 = 1
        self.prev_model = None

    @property
    def name(self) -> str:
        return "lwf"

    def record_state(self) -> None:
        self.prev_model = copy.deepcopy(self.model)

    def _distillation_loss(
        self, current_out: torch.FloatTensor, prev_out: torch.FloatTensor
    ) -> torch.FloatTensor:
        log_p = torch.log_softmax(current_out / self.temp, dim=1)
        q = torch.softmax(prev_out / self.temp, dim=1)
        result = torch.nn.KLDivLoss(reduction="batchmean")(log_p, q)
        return result
    
    def lwf_loss(
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
            )

        return dist_loss

    def observe(self, 
            x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, 
            not_aug_x: Optional[torch.FloatTensor] = None
        ) -> torch.FloatTensor:

        features = self.model.forward_backbone(x)

        pred = self.model.forward_head(features, y, t)
        inc_loss = self.loss(pred, y)

        lwf_loss = self.lwf_loss(
            features=features, data=x, y=y, current_model=self.model, current_task=t
        )

        loss = inc_loss + self.lambda_0 * lwf_loss
        self.update(loss)

        return loss

    def lwf_supconloss(
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
            )

        return dist_loss

    def supcon_observe(self, 
            x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor, 
            not_aug_x: Optional[torch.FloatTensor] = None
        ) -> torch.FloatTensor:
        assert isinstance(self.loss, SupConLoss), "supcon_observe function must have SupconLoss"
        bsz = y.shape[0]

        # compute current model supcon loss
        backbone_features = self.model.forward_backbone(x)
        features = self.model.forward_head(backbone_features)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # [512, 2, 128]
        inc_loss = self.loss(features, y)

        lwf_supconloss = self.lwf_supconloss(
            features=backbone_features, data=x, y=y, current_model=self.model, current_task=t
        )

        loss = inc_loss + self.lambda_0 * lwf_supconloss
        self.update(loss)

        return loss

    def on_task_end(self) -> None:
        super().on_task_end()
        self.record_state()
