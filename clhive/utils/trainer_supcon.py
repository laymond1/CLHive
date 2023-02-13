from typing import Optional, Union, List
import copy
import os
import time
from argparse import Namespace

import torch
from torch.utils.data import DataLoader

from torch import autograd

from .evaluators import BaseEvaluator
from .generic import warmup_learning_rate
from ..loggers import BaseLogger, Logger
from ..methods import BaseMethod
from ..models import ContinualModel, ContinualAngularModel, SupConLoss
from ..scenarios import ClassIncremental, TaskIncremental


class SupConTrainer:
    def __init__(
        self,
        opt: Namespace,
        method: BaseMethod,
        scenario: Union[ClassIncremental, TaskIncremental],
        n_epochs: int,
        evaluator: Optional[BaseEvaluator] = None,
        logger: Optional[BaseLogger] = None,
        device: Optional[torch.device] = None,
    ) -> "SupConTrainer":

        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.opt = opt
        self.agent = method.to(self.device)
        self.scenario = scenario
        self.n_epochs = n_epochs
        self.evaluator = evaluator

        if logger is None:
            logger = Logger(n_tasks=self.scenario.n_tasks)
        self.logger = logger

    def _train_task(self, task_id: int, train_loader: DataLoader):

        self.agent.train()
        self.agent.on_task_start()

        start_time = time.time()

        print(f"\n>>> Task #{task_id} --> Model Training")

        for epoch in range(self.n_epochs):
            # adjust learning rate

            for idx, (x, y, t, not_aug_x) in enumerate(train_loader):
                # concat images
                x = torch.cat([x[0], x[1]], dim=0)
                x, y, t, not_aug_x = x.to(self.device), y.to(self.device), t.to(self.device), not_aug_x.to(self.device)

                # Do we need warm-up learning rate in SupCon Paper?
                warmup_learning_rate(args=self.opt, 
                                     epoch=self.n_epochs, 
                                     batch_id=idx, 
                                     total_batches=len(train_loader), 
                                     optimizer=self.agent.optim)

                # Supcon loss
                self.agent.loss = SupConLoss(temperature=0.05, contrast_mode='all')

                loss = self.agent.supcon_observe(x, y, t, not_aug_x)

                print(
                    f"Epoch: {epoch + 1} / {self.n_epochs} | {idx} / {len(train_loader)} - Loss: {loss}",
                    end="\r",
                )

        print(f"Task {task_id}. Time {time.time() - start_time:.2f}")

    def set_evaluator(self, evaluator: BaseEvaluator):
        self.evaluator = evaluator

    def on_task_start(self, task_id: int):
        """ """
        pass

    def on_task_end(self, task_id: int):
        """_summary_

        Args:
            task_id (int): _description_
        """
        # Agent on_task_finished
        finished_task_id = self.agent._current_task_id
        self.agent.on_task_end()

        # Launch evaluators : update code that it can use multi-evaluators
        if self.evaluator is not None:
            if isinstance( self.evaluator, List):
                for evaluator in self.evaluator:
                    evaluator.fit(current_task_id=finished_task_id)
            else:
                self.evaluator.fit(current_task_id=finished_task_id)

    def on_training_start(self):
        """ """
        pass

    def on_training_end(self):
        """ """
        pass

    def fit(self):
        """ """
        self.on_training_start()

        for task_id, train_loader in enumerate(self.scenario):
            self.on_task_start(task_id)

            self._train_task(task_id=task_id, train_loader=train_loader)

            self.on_task_end(task_id)

        self.on_training_end()
