from typing import Dict, List, Optional, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import auto_model
from . import register_model
from ..config import Config
from ..scenarios import ClassIncremental, TaskIncremental


class BaseFace(nn.Module):
    """
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        s: float, 
        m: float, 
        **kwargs
    ) -> "BaseFace":
        """_summary_

        Args:
            input_size (_type_): _description_
            output_size (_type_): _description_
            s (float): _description_
            m (float): _description_

        Returns:
            BaseFace: _description_
        """    
        super(BaseFace, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.s = s
        self.m = m

        self.w = nn.Parameter(torch.Tensor(input_size, output_size))
        nn.init.xavier_normal_(self.w)

    @property
    def name(self) -> str:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, x):
        """ """
        pass

    def forward(self, x, y):
        """ """
        pass


@register_model("arcface")
class ArcFace(BaseFace):
    """ reference: <Additive Angular Margin Loss for Deep Face Recognition>
    """
    def __init__(self, input_size, output_size, s=64., m=0.5, **kwargs):
        super().__init__(input_size, output_size, s, m, **kwargs)

    @torch.no_grad()
    def predict(self, x):
        self.w.data = F.normalize(self.w.data, dim=0)
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
        theta_m.clamp_(1e-5, 3.14159)
        d_theta = torch.cos(theta_m) - cos_theta

        logits = self.s * (cos_theta + d_theta)
        return logits

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
            theta_m.scatter_(1, y.view(-1, 1), self.m, reduce='add')
            theta_m.clamp_(1e-5, 3.14159)
            d_theta = torch.cos(theta_m) - cos_theta

        logits = self.s * (cos_theta + d_theta)
        # loss = F.cross_entropy(logits, y)

        return logits


@register_model("cosface")
class CosFace(BaseFace):
    """reference1: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
       reference2: <Additive Margin Softmax for Face Verification>
    """
    def __init__(self, input_size, output_size, s=64., m=0.35, **kwargs):
        super().__init__(input_size, output_size, s, m, **kwargs)

    @torch.no_grad()
    def predict(self, x):
        self.w.data = F.normalize(self.w.data, dim=0)
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        d_theta = torch.zeros_like(cos_theta)

        logits = self.s * (cos_theta + d_theta)

        return logits

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            d_theta = torch.zeros_like(cos_theta)
            d_theta.scatter_(1, y.view(-1, 1), -self.m, reduce='add')

        logits = self.s * (cos_theta + d_theta)
        # loss = F.cross_entropy(logits, y)

        return logits


@register_model("gasoftmax")
class GAsoftmax(BaseFace):
    """ reference: <Deep Hyperspherical Learning> in NIPS 2017"
        Section 3 in the paper: https://arxiv.org/abs/1711.03189
    """
    def __init__(self, input_size, output_size, s=30., m=1.5, **kwargs):
        super().__init__(input_size, output_size, s, m, **kwargs)

    @torch.no_grad()
    def predict(self, x):
        # weight normalization
        self.w.data = F.normalize(self.w.data, dim=0)

        # cos_theta and d_theta
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        #with torch.no_grad():
        m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
        m_theta_ori = m_theta

        confid = -0.63662 * (m_theta_ori) + 1.
        logits = self.s * (confid)

        return logits

    def forward(self, x, y):
        # weight normalization
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # cos_theta and d_theta
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        #with torch.no_grad():
        m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
        m_theta_ori = m_theta
        with torch.no_grad():
            m_theta.scatter_(
                1, y.view(-1, 1), self.m, reduce='multiply',
            )
            m_theta_offset = m_theta - m_theta_ori

        confid = -0.63662 * (m_theta_ori + m_theta_offset ) + 1.
        logits = self.s * (confid)
        # loss = F.cross_entropy(logits, y)

        return logits


@register_model("sphereface")
class SphereFace(BaseFace):
    """ reference: <SphereFace: Deep Hypersphere Embedding for Face Recognition>"
        It also used characteristic gradient detachment tricks proposed in
        <SphereFace Revived: Unifying Hyperspherical Face Recognition>.
    """
    def __init__(self, input_size, output_size, s=30., m=1.5, **kwargs):
        super().__init__(input_size, output_size, s, m, **kwargs)

    @torch.no_grad()
    def predict(self, x):
        self.w.data = F.normalize(self.w.data, dim=0)

        # cos_theta and d_theta
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
        k = (m_theta / math.pi).floor()
        sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
        phi_theta = sign * torch.cos(m_theta) - 2. * k
        d_theta = phi_theta - cos_theta

        logits = self.s * (cos_theta + d_theta)

        return logits

    def forward(self, x, y):
        # weight normalization
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # cos_theta and d_theta
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
            m_theta.scatter_(
                1, y.view(-1, 1), self.m, reduce='multiply',
            )
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
            phi_theta = sign * torch.cos(m_theta) - 2. * k
            d_theta = phi_theta - cos_theta

        logits = self.s * (cos_theta + d_theta)
        # loss = F.cross_entropy(logits, y)

        return logits


class ContinualAngularModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        heads: Union[nn.ModuleList, nn.Module],
        scenario: Union[ClassIncremental, TaskIncremental],
    ) -> "ContinualAngularModel":

        super(ContinualAngularModel, self).__init__()

        self.backbone = backbone
        self.scenario = scenario
        self.heads = nn.ModuleList(heads)

        if isinstance(scenario, TaskIncremental):
            msg = (
                "Number of `heads` should be equal to `n_tasks`"
                + "in TaskIncremental scenario, "
                + f"expected {scenario.n_tasks} nn.Modules but received {len(self.heads)}."
            )
            assert len(self.heads) == scenario.n_tasks, msg

    @classmethod
    def auto_model(
        cls,
        backbone_name: str,
        scenario: Union[ClassIncremental, TaskIncremental],
        image_size: Optional[int] = None,
        head_name: Optional[str] = "arcface",
    ) -> "ContinualAngularModel":

        backbone = auto_model(name=backbone_name, input_size=image_size)

        if isinstance(scenario, TaskIncremental):
            heads = [
                auto_model(
                    name=head_name,
                    input_size=backbone.output_dim,
                    output_size=scenario.loader.sampler.cpt,
                )
                for t in range(scenario.n_tasks)
            ]
        else:
            heads = [
                auto_model(
                    name=head_name,
                    input_size=backbone.output_dim,
                    output_size=scenario.n_classes,
                )
            ]

        return cls(backbone, heads, scenario)

    @classmethod
    def from_config(cls, config: Config) -> "ContinualAngularModel":
        """Instantiates a Model from a configuration.

        Args:
            config (Dict): A configuration for the Model.

        Returns:
            A torch.nn.Module instance.
        """

        return cls(*args, **kwargs)

    def set_heads(self, heads: nn.ModuleDict) -> None:
        self.heads = heads

    def add_head(self, name: str, head: nn.Module) -> None:
        self.heads.update({name: head})

    def forward_backbone(self, x) -> torch.Tensor:
        return self.backbone(x)
    
    #TODO Only ClassIncremental setting is clear, check TaskIncremental setting
    def forward_head(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if (t is None) or (not isinstance(self.scenario, TaskIncremental)):
            return self.heads[0](x, y)

        pred = torch.zeros(
            x.size(0),
            self.heads[0].output_size,
            device=next(self.backbone.parameters()).device,
        )
        tasks = t.unique().tolist()
        for task in tasks:
            idx = t == task
            pred[idx] = self.heads[task](x[idx, ...], y[idx, ...]) # y is not verified

        return pred

    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform computation of blocks in the order define in get_blocks.
        """
        if isinstance(self.scenario, TaskIncremental):
            assert t.max() < len(
                self.heads
            ), f"head number {t} does not exist in `ContinualAngularModel.heads`"

        x = self.forward_backbone(x)
        x = self.forward_head(x, y, t)
        return x

    def predict_head(self, 
        x: torch.Tensor, 
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if (t is None) or (not isinstance(self.scenario, TaskIncremental)):
            return self.heads[0].predict(x)

        pred = torch.zeros(
            x.size(0),
            self.heads[0].output_size,
            device=next(self.backbone.parameters()).device,
        )
        tasks = t.unique().tolist()
        for task in tasks:
            idx = t == task
            pred[idx] = self.heads[task].predict(x[idx, ...])

        return pred

    def predict(
        self, 
        x: torch.FloatTensor, 
        t: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Perform computation of blocks in the order define in get_blocks.
        """
        if isinstance(self.scenario, TaskIncremental):
            assert t.max() < len(
                self.heads
            ), f"head number {t} does not exist in `ContinualModel.heads`"

        x = self.forward_backbone(x)

        x = self.predict_head(x, t)
        return x
