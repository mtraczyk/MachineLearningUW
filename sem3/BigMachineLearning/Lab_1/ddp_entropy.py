import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP


class NN(nn.Module):  # inherits nn.Module

    def __init__(self, input_size, num_classes):  # input size = 28x28 = 784 for mnist
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


batch_size = 64
input_size = 784
num_classes = 10
learning_rate = 0.001
num_epochs = 5

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cpu')  # For this example cpu will be enough

# The only changes needed to distribute learning
# To initialize process group we need to specify: communication backend, rank (id of the process), world size (number of processes)
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
RANK = int(os.environ['SLURM_PROCID']) #int(os.environ['RANK']) 
# WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
# RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
# Pytorch expects two environment variables, let's just set them here for now
os.environ['MASTER_ADDR'] = os.environ['MASTER_ADDR']
os.environ['MASTER_PORT'] = os.environ['MASTER_PORT']
dist.init_process_group("gloo", rank=RANK, world_size=WORLD_SIZE)
model = DDP(NN(input_size=input_size, num_classes=num_classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()

for epoch in range(num_epochs):
    print(f'Epoch: {epoch}')
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        # print(data.shape)  # => [64 , 1, 28, 28] => 64 : num_images, 1 -> num_channels, (28,28): (height, width)
        data = data.reshape(data.shape[0], -1)  # Flatten
        # if epoch == 0 and batch_idx == 0:
        #    print(data.shape)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

end_time = time.time()
print(f"Learning took {end_time - start_time}s")


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Accuracy on training data")
    else:
        print("Accuracy on testing data")

    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100: .2f}')
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
