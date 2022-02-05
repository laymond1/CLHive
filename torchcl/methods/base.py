from cmath import log
import copy
from statistics import mode
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis as FCA

from torchcl.models import ModelWrapper
from torchcl.data.transforms import BaseTransform
from torchcl.utils import get_optimizer


class BaseMethod(nn.Module):
    """[summary]

    """

    def __init__(
        self,
        model : ModelWrapper,
        logger,
        config,
        transform: Optional[BaseTransform] = BaseTransform,
        optim: Optional[torch.optim] = None,
    ) -> None:

        super(BaseMethod, self).__init__()

        self.model = model
        self.logger = logger
        self.transform = transform
        self.config = config

        if optim is None:
            optim = get_optimizer(self.config)
        self.optim = optim

    
    @property
    def name(self):
        raise NotImplementedError

    @property
    def one_sample_flop(self):
        """[summary]
        """
        if not hasattr(self, '_train_cost'):
            input = torch.FloatTensor(size=(1,) + self.config.input_size).to(self.device)
            flops = FCA(self.model, input)
            self._train_cost = flops.total() / 1e6 # MegaFlops

        return self._train_cost

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def update(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def observe(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def on_task_start(self):
        raise NotImplementedError

    def on_task_end(self):
        raise NotImplementedError

    
