from typing import Any, Callable, Dict, Optional, Sequence, Union

from PIL import Image
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from . import register_dataset
from .continual_dataset import ContinualDataset


class CIFARDataset(ContinualDataset):

    _DATA_TYPE = None
    _DEFAULT_N_TASKS = None
    _MEAN = (0.4914, 0.4822, 0.4465)
    _STD = (0.2470, 0.2435, 0.2615)
    _IMAGE_SIZE = 32

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        normalize_targets_per_task: Optional[bool] = False,
        train: Optional[bool] = True,
        download: Optional[bool] = True,
    ) -> None:
        
        assert self._DATA_TYPE in [
            "cifar10",
            "cifar100",
        ], "CIFARDataset must be subclassed and a valid _DATA_TYPE provided"

        if self._DATA_TYPE == "cifar10":
            dataset = CIFAR10(root, train=train, download=download)
        if self._DATA_TYPE == "cifar100":
            dataset = CIFAR100(root, train=train, download=download)

        super().__init__(
            dataset=dataset,
            transform=transform,
            normalize_targets_per_task=normalize_targets_per_task,
        )
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])

    @classmethod
    def from_config(cls, config: Dict[str, Any], train: bool) -> "CIFARDataset":
        """Instantiates a CIFARDataset from a configuration.
        Args:
            config: A configuration for a CIFARDataset.
                See :func:`__init__` for parameters expected in the config.
        Returns:
            A CIFARDataset instance.
        """

        root = config.get("root")
        train = train
        transform_config = config.get("transform")
        download = config.get("download")

        transform = get_transform(transform_config)
        return cls(root=root, transform=transform, train=train, download=True)
    
    def __getitem__(self, index: int):
        assert index >= 0 and index < len(
            self.dataset
        ), f"Provided index ({index}) is outside of dataset range."

        sample = self.dataset[index]
        data, targets = sample

        # to return a PIL Image
        original_img = data.copy()

        not_aug_img = self.not_aug_transform(original_img)

        data = self.transform(data)

        if self.normalize_targets_per_task and self.n_classes_per_task:
            targets -= self._current_task * self.n_classes_per_task

        return data, targets, self._current_task, not_aug_img


@register_dataset("cifar10")
class SplitCIFAR10(CIFARDataset):
    _DATA_TYPE = "cifar10"
    _DEFAULT_N_TASKS = 5


@register_dataset("cifar100")
class SplitCIFAR100(CIFARDataset):
    _DATA_TYPE = "cifar100"
    _DEFAULT_N_TASKS = 20       
    