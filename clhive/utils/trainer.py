from typing import Optional, Union, List
import copy
import os
import time
import numpy as np
from argparse import Namespace

import torch
from torch.utils.data import DataLoader

from .evaluators import BaseEvaluator
from ..loggers import BaseLogger, Logger, AverageMeter, create_if_not_exists
from ..methods import BaseMethod
from ..models import ContinualModel
from ..scenarios import ClassIncremental, TaskIncremental


class Trainer:
    def __init__(
        self,
        opt: Namespace,
        method: BaseMethod,
        scenario: Union[ClassIncremental, TaskIncremental],
        n_epochs: int,
        evaluator: Union[BaseEvaluator, List] = None,
        logger: Optional[BaseLogger] = None,
        device: Optional[torch.device] = None,
    ) -> "Trainer":

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

        task_time = time.time()
        
        self.logger.write_txt(f"\n>>> Task #{task_id} --> Model Training")

        for epoch in range(self.n_epochs):
            # adjust learning rate

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()

            end = time.time()
            for idx, (x, y, t, not_aug_x) in enumerate(train_loader):
                # batch data load time
                data_time.update(time.time() - end)

                x, y, t = x.to(self.device), y.to(self.device), t.to(self.device)
                loss = self.agent.observe(x, y, t, not_aug_x)
                # update metric
                losses.update(loss.item(), y.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
          
                # if (idx + 1) % 5 == 0:
                # print('Train: [{0}][{1}/{2}]\t'
                #     'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #     'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                #     'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                #     epoch, idx + 1, len(train_loader), batch_time=batch_time,
                #     data_time=data_time, loss=losses), end='\r')
                # sys.stdout.flush()

            self.logger.write_txt('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    epoch+1, idx+1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

        self.logger.write_txt(f"Task {task_id}. Time {time.time() - task_time:.2f}")

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
            if isinstance(self.evaluator, List):
                for evaluator in self.evaluator:
                    tasks_accs = evaluator.fit(current_task_id=finished_task_id, logger=self.logger)
                    self.d[evaluator.name].append(tasks_accs)

            else:
                tasks_accs = self.evaluator.fit(current_task_id=finished_task_id, logger=self.logger)
                self.d[evaluator.name].append(tasks_accs)

    def on_training_start(self):
        """ """
        self.d = dict()
        self.d['Base'] = []
        self.d['LP'] = []
        self.d['Rep'] = []
        # pass

    def on_training_end(self):
        """ turn dict into dataframe"""
        self.d['Base'] = np.array(self.d['Base'])
        self.d['LP'] = np.array(self.d['LP'])
        self.d['Rep'] = np.array(self.d['Rep'])

        self.logger.store_results(self.d, self.opt.save_path)
        # pass

    def fit(self):
        """ """
        self.on_training_start()

        for task_id, train_loader in enumerate(self.scenario):
            self.on_task_start(task_id)

            self._train_task(task_id=task_id, train_loader=train_loader)

            self.on_task_end(task_id)

        self.on_training_end()
