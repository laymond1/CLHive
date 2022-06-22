## Overview
**CLHive** is a codebase on top of [PyTorch](https://pytorch.org) for Continual Learning research. It provides the components necessary to run CL experiments, for both task-incremental and class-incremental settings. It is designed to be readable and easily extensible, to allow users to quickly run and experiment with their own ideas.


## How To Use

With *clhive* you can use latest continual learning methods in a modular way using the full power of PyTorch. Experiment with different backbones, models and loss functions. The framework has been designed to be easy to use from the ground up.

### Dependencies

Lightly requires **Python 3.6+**.

- fvcore
- hydra-core>=1.0.0
- numpy>=1.18.1
- pytorch
- rich
- torchvision
- wandb

### Quick Start

```python
from clhive.data import SplitCIFAR10
from clhive.scenarios import ClassIncremental
from clhive.models import ContinualModel
from clhive.methods import auto_method

dataset = SplitCIFAR10(root="../cl-datasets/data/")
scenario = ClassIncremental(dataset=dataset, n_tasks=5, batch_size=32)

print(
  f"Number of tasks: {scenario.n_tasks} | Number of classes: {scenario.n_classes}"
)

model = ContinualModel.auto_model("resnet18", scenario, image_size=32)
agent = auto_method(
    name="finetuning", model=model, optim=SGD(model.parameters(), lr=0.01)
)

for task_id, train_loader in enumerate(scenario):
    for x, y, t in train_loader:
        # Do your cool stuff here
        loss = agent.observe(x, y, t)
        ...
```

### Command-Line Interface

Lightly is accessible also through a command-line interface (CLI). To train a ER model on Tiny-ImageNet you can simply run the following command:

```
python main.py ...
```

<details>
  <summary>More CLI examples:</summary>
  
Train CLIP with ViT-base on COCO Captions dataset:

```
python main.py data=coco model/vision_model=vit-b  model/text_model=vit-b
```

</details>



## Terminology

Below you can see a schematic overview of the different concepts present in the *clhive* Python package.



## Reading The Commits
Here is a reference to what each emoji in the commits means:

* 📎 : Some basic updates.
* ♻️ : Refactoring.
* 💩 : Bad code, needs to be revised!
* 🐛 : Bug fix.
* 💡 : New feature.
* ⚡ : Performance Improvement.
