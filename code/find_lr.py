# Python 3.10.4
# torch==1.11.0+cu113
# torchvision==0.12.0+cu113
# matplotlib==3.5.1
# pip install accelerate==0.6.2

from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.datasets import CIFAR10
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import ipdb

from collections import OrderedDict
from time import perf_counter

DOWNLOAD_DIR = Path(".").resolve() / "cifar10"

class ConvReluBN(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, padding, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, padding=padding, stride=stride, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class TestCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 7),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5,5)),
            nn.Flatten(),
            nn.Linear(5*5*32, n_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ("block1", ConvReluBN(3, 16, 7, 3)),
            ("block2", self._block(16, 32)),
            ("block3", self._block(32, 64)),
            ("block4", self._block(64, 128)),
            ("pool", nn.AdaptiveAvgPool2d((1, 1))),
            ("flatten", nn.Flatten()),
            ("head", nn.Linear(128, n_classes))
        ]))

    def _block(self, ch_in, ch_out):
        return nn.Sequential(ConvReluBN(ch_in, ch_in, 3, 1), ConvReluBN(ch_in, ch_out, 3, 1), ConvReluBN(ch_out, ch_out, 2, 0, 2))

    def forward(self, x):
        x = self.model(x)
        return x

def elapsed_time():
    return f"{perf_counter() - START_TIME:.1f}"


if __name__ == "__main__":
    START_TIME = perf_counter()

    BATCH_SIZE = 1024
    EPOCHS = 3
    LEARNING_RATE = 0.1
    PLOT_LR = False

    TEST_CNN = False
    OVERFIT_BATCH = False
    FIND_LR = False
    BUG_OPT = False
    BUG_VAL = False

    dataset_train = CIFAR10(root=DOWNLOAD_DIR, train=True, transform=TF.to_tensor, target_transform=None, download=True)
    dataset_val = CIFAR10(root=DOWNLOAD_DIR, train=False, transform=TF.to_tensor, target_transform=None, download=True)

    model = CNN(10)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    steps_per_epoch = len(dataloader_train)
    total_steps = steps_per_epoch * EPOCHS
    lr_scheduler = OneCycleLR(optimizer, LEARNING_RATE, total_steps=total_steps)

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    model, optimizer, dataloader_train, dataloader_val, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader_train, dataloader_val, lr_scheduler
    )

    if OVERFIT_BATCH:
        batch_to_overfit = next(dataloader_train.__iter__())
        dataloader_train = [batch_to_overfit] * steps_per_epoch

    train_losses = []
    val_losses = []
    learning_rates = []
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS} ({elapsed_time()})")
        count = 0
        avg_loss = 0.0
        model = model.train()
        for images, classes in dataloader_train:
            images = images.to(device)
            classes = classes.to(device)

            if not BUG_OPT:
                optimizer.zero_grad()

            output = model(images)
            loss = F.cross_entropy(output, classes)
            accelerator.backward(loss)

            if not BUG_OPT:
                optimizer.step()
            learning_rates.append(lr_scheduler.get_last_lr()[0])
            lr_scheduler.step()

            train_loss = loss.detach().cpu().numpy()

            count += 1
            avg_loss += train_loss
            train_losses.append(train_loss)

            # TODO: what's a good guaranteed method to clear a line?
            print(f"Train loss = {avg_loss/count:.4f}, step={count}/{steps_per_epoch}", end="                          \r")
        print(f"Train loss = {avg_loss/count:.4f}, step={count}/{steps_per_epoch} ({elapsed_time()}).")

        count = 0
        avg_loss = 0.0
        model = model.eval()
        if BUG_VAL:
            dataloader_val = dataloader_train
        for images, classes in dataloader_val:
            images = images.to(device)
            classes = classes.to(device)

            output = model(images)
            loss = F.cross_entropy(output, classes)

            val_loss = loss.detach().cpu().numpy()

            count += 1
            avg_loss += val_loss
        avg_loss /= count
        val_losses.append(avg_loss)
        print(f"Val. loss = {avg_loss:.4f} ({elapsed_time()}).")

    # TODO: save plots
    if PLOT_LR:
        fig, axs = plt.subplots(2)
        axs[0].plot(range(1, total_steps + 1), train_losses, label="Train loss")
        axs[0].plot(range(1, total_steps + 2, steps_per_epoch), [train_losses[0], *val_losses], label="Val. loss")
        axs[1].plot(range(1, total_steps + 1), learning_rates)
        plt.legend()
        plt.show()
    else:
        plt.plot(range(1, total_steps + 1), train_losses, label="Train loss")
        plt.plot(range(1, total_steps + 2, steps_per_epoch), [train_losses[0], *val_losses], label="Val. loss")
        plt.legend()
        plt.show()
