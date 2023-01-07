import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .base import BaseEvaluator
from ...loggers import BaseLogger, Logger
from ...methods import BaseMethod
from ...models import ContinualModel, LinearClassifier
from ...scenarios import ClassIncremental, TaskIncremental, RepresentationIncremental

#TODO 
class RepresentationEvaluator(BaseEvaluator):
    def __init__(
        self,
        method: BaseMethod,
        eval_scenario: Union[ClassIncremental, TaskIncremental],
        logger: Optional[BaseLogger] = None,
        device: Optional[torch.device] = None,
    ) -> "RepresentationEvaluator":
        """_summary_

        Args:
            method (BaseMethod): _description_
            eval_scenario (Union[ClassIncremental, TaskIncremental]): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            device (Optional[torch.device], optional): _description_. Defaults to None.

        Returns:
            ContinualEvaluator: _description_
        """
        super().__init__(method, eval_scenario, logger, device)

    @torch.no_grad()
    def _evaluate(self, task_id: int) -> List[float]:
        """_summary_

        Args:
            task_id (int): _description_

        Returns:
            _type_: _description_
        """
        self.agent.eval()
        mb_size = self.eval_scenario.batch_size
        # need feat dim
        dim = 256
        feats = torch.zeros([self.eval_scenario.n_samples, 2, dim], dtype=torch.float32).to(self.device)

        for idx, (query_x, retrieval_x, y) in enumerate(self.eval_scenario.loader):
            query_x, retrieval_x, y = query_x.to(self.device), retrieval_x.to(self.device), y.to(self.device)

            qeury_feat = self.agent.model.backbone(query_x)
            # query_x = torch.flip(query_x, [3]) # flip
            # qeury_feat += self.agent.model.backbone(query_x)
            retrieval_feat = self.agent.model.backbone(retrieval_x)
            # retrieval_x = torch.flip(retrieval_x, [3]) # flip
            # retrieval_feat += self.agent.model.backbone(retrieval_x)

            feats[(idx*mb_size):(idx+1)*mb_size, 0, :] = qeury_feat
            feats[(idx*mb_size):(idx+1)*mb_size, 1, :] = retrieval_feat

        results = self.eval_scenario.evaluate(feats.cpu())
        results = dict(results)
        metric = 'ACC'
        
        return results[metric]

    def on_eval_start(self):
        """ """
        pass

    def on_eval_end(self, tasks_acc: float, current_task_id: int) -> None:
        """Representation Evluation Setting is not divided into tasks.
           Just one task of pair-wise classification.
        """
        print(
            "\n",
            f"Representation | Observed  ACC: {tasks_acc:.2f} |"
        )

        return tasks_acc

    def fit(self, current_task_id: int) -> None:
        """_summary_

        Args:
            current_task_id (int): _description_
        """
        self.on_eval_start()

        tasks_acc = self._evaluate(task_id=current_task_id)

        self.on_eval_end(tasks_acc=tasks_acc, current_task_id=current_task_id)