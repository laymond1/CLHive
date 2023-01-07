from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from . import auto_model
from ..config import Config
from ..scenarios import ClassIncremental, TaskIncremental


class ContinualModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        heads: Union[nn.ModuleList, nn.Module],
        scenario: Union[ClassIncremental, TaskIncremental],
    ) -> "ContinualModel":

        super(ContinualModel, self).__init__()

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
        head_name: Optional[str] = "linear",
    ) -> "ContinualModel":

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
    def from_config(cls, config: Config) -> "ContinualModel":
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

    def forward_head(
        self, 
        x: torch.Tensor, 
        y: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if (t is None) or (not isinstance(self.scenario, TaskIncremental)):
            return self.heads[0](x)

        pred = torch.zeros(
            x.size(0),
            self.heads[0].output_size,
            device=next(self.backbone.parameters()).device,
        )
        tasks = t.unique().tolist()
        for task in tasks:
            idx = t == task
            pred[idx] = self.heads[task](x[idx, ...])

        return pred

    def forward(
        self, 
        x: torch.Tensor, 
        y: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform computation of blocks in the order define in get_blocks.
        """
        if isinstance(self.scenario, TaskIncremental):
            assert t.max() < len(
                self.heads
            ), f"head number {t} does not exist in `ContinualModel.heads`"

        x = self.forward_backbone(x)
        x = self.forward_head(x, t)
        return x

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
        x = self.forward_head(x, t)
        return x
