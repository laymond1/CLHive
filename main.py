import torch
from torch.optim import SGD

from clhive.data import SplitCIFAR10
from clhive.scenarios import ClassIncremental, TaskIncremental
from clhive.models import ContinualModel
from clhive.methods import auto_method

dataset = SplitCIFAR10(root="../cl-datasets/data/")
scenario = TaskIncremental(dataset=dataset, n_tasks=5, batch_size=32, n_workers=6)

print(f"Number of tasks: {scenario.n_tasks} | Number of classes: {scenario.n_classes}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ContinualModel.auto_model("resnet18", scenario, image_size=128).to(device)

agent = auto_method(
    name="finetuning", model=model, optim=SGD(model.parameters(), lr=0.01)
)

for task_id, train_loader in enumerate(scenario):
    for x, y, t in train_loader:
        # Do your cool stuff here
        x, y, t = x.to(device), y.to(device), t.to(device)
        loss = agent.observe(x, y, t)

        print(f"Task: {task_id} - Loss: {loss.item()}", end="\r")
