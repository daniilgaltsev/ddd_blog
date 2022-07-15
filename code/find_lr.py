# Python 3.10.4
# torch==1.11.0+cu113
# torchvision==0.12.0+cu113
# matplotlib==3.5.1
# accelerate==0.6.2
# Need for colab and probably others: pip install accelerate

from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, MultiplicativeLR
from torchvision.datasets import CIFAR10
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from collections import OrderedDict
from time import perf_counter

DOWNLOAD_DIR = Path(".").resolve() / "cifar10"


def elapsed_time():
    return f"{perf_counter() - START_TIME:.1f}"


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

class Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.pre_residual = nn.Sequential(
            ConvReluBN(ch_in, ch_in, 3, 1),
            ConvReluBN(ch_in, ch_in, 3, 1)
        )
        self.post_residual = ConvReluBN(ch_in, ch_out, 2, 0, 2)

    def forward(self, x):
        inp = x
        x = self.pre_residual(x)
        x += inp
        x = self.post_residual(x)
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
        return Block(ch_in, ch_out)

    def forward(self, x):
        x = self.model(x)
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


def train(
    model, dataloader_train, dataloader_val, accelerator, optimizer, lr_scheduler, 
    train_losses, val_losses, learning_rates, steps_per_epoch
    ):
    device = accelerator.device
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS} ({elapsed_time()})")
        count = 0
        correct = 0
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
            
            predicted = torch.argmax(output.detach(), 1)
            correct += torch.sum(predicted == classes).cpu().item()

            if not BUG_OPT:
                optimizer.step()
            learning_rates.append(lr_scheduler.get_last_lr()[0])
            lr_scheduler.step()

            train_loss = loss.detach().cpu().item()

            count += 1
            avg_loss += train_loss
            train_losses.append(train_loss)

            # TODO: what's a good guaranteed method to clear a line?
            print(f"Train loss={avg_loss/count:.4f}, train acc.={correct/(count*BATCH_SIZE):.4f} step={count}/{steps_per_epoch}", end="                          \r")
        print(f"Train loss={avg_loss/count:.4f}, train acc.={correct/(count*BATCH_SIZE):.4f}, step={count}/{steps_per_epoch} ({elapsed_time()}).")

        count = 0
        correct = 0
        avg_loss = 0.0
        model = model.eval()
        if BUG_VAL:
            dataloader_val = dataloader_train
        for images, classes in dataloader_val:
            images = images.to(device)
            classes = classes.to(device)

            output = model(images)
            loss = F.cross_entropy(output, classes)

            predicted = torch.argmax(output.detach(), 1)
            correct += torch.sum(predicted == classes).cpu().item()

            val_loss = loss.detach().cpu().item()

            count += 1
            avg_loss += val_loss
        avg_loss /= count
        val_losses.append(avg_loss)
        print(f"Val. loss={avg_loss:.4f}, val. acc.={correct/(count*BATCH_SIZE):.4f} ({elapsed_time()}).")


def save_plot(name):
    if SAVE_PLOTS: plt.savefig(f"{name}_test{TEST_CNN}_overfit{OVERFIT_BATCH}_find{FIND_LR}_opt{BUG_OPT}_val{BUG_VAL}")


if __name__ == "__main__":
    START_TIME = perf_counter()

    BATCH_SIZE = 1024
    EPOCHS = 10
    LEARNING_RATE = 0.024 # 0.01 is random initial, 0.024 for the CNN, 0.0022 for the TestCNN
    # <lr> - <train acc>/<val. acc> # Default lr = 0.001
    # CNN:     0.240 - 87.8/76.5, 0.0240 - 93.0/77.1
    # TestCNN: 0.022 - 41.7/40.8, 0.0022 - 50.4/49.0
    # |Train/Val. acc|    1.0         0.1         0.01       0.001       0.0001       0.00001
    # CNN              10.0/9.8    92.0/76.9   91.4/75.3   77.6/67.4    51.4/49.2    29.5/28.8
    # TestCNN           9.8/9.8     9.8/9.8    53.4/51.5   46.5/45.2    32.7/32.1    15.9/15.5
    # __LEARNING RATE FINDER RESULTS__
    # For 1.5 multiplier
    # Results for TestCNN: [0.0131, 0.0131, 0.0131, 0.0196, 0.0131]
    # Results for CNN:     [0.0087, 0.0058, 0.0197, 0.0295, 0.0131]
    # For 1.2 multiplier
    # Results for TestCNN: [0.0310, 0.0060, 0.0060, 0.0050, 0.0086]
    # Results for CNN:     [0.0104, 0.0124, 0.0258, 0.1917, 0.0050]
    # For 1.1 multiplier
    # Results for TestCNN: [0.0168, 0.0168, 0.0153, 0.0126, 0.0223]
    # Results for CNN:     [0.1130, 0.2662, 0.2420, 0.1130, 0.2420]
    PLOT_LR = False
    SAVE_PLOTS = True

    TEST_CNN = False
    OVERFIT_BATCH = False
    FIND_LR = False
    BUG_OPT = False
    BUG_VAL = False

    if FIND_LR:
        LEARNING_RATE = 1e-7

    dataset_train = CIFAR10(root=DOWNLOAD_DIR, train=True, transform=TF.to_tensor, target_transform=None, download=True)
    dataset_val = CIFAR10(root=DOWNLOAD_DIR, train=False, transform=TF.to_tensor, target_transform=None, download=True)

    n_classes = 10
    if TEST_CNN:
        model = TestCNN(n_classes)
    else:
        model = CNN(n_classes)
    print(model)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    steps_per_epoch = len(dataloader_train)
    total_steps = steps_per_epoch * EPOCHS
    lr_scheduler = OneCycleLR(optimizer, LEARNING_RATE, total_steps=total_steps)
    if FIND_LR:
        lr_scheduler = MultiplicativeLR(optimizer, lambda idx: 1.1)

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

    if not FIND_LR:
        train(
            model, dataloader_train, dataloader_val, accelerator, optimizer, lr_scheduler, 
            train_losses, val_losses, learning_rates, steps_per_epoch
        )

        if PLOT_LR:
            fig, axs = plt.subplots(2)
            axs[0].plot(range(1, total_steps + 1), train_losses, label="Train loss")
            axs[0].plot(range(1, total_steps + 2, steps_per_epoch), [train_losses[0], *val_losses], label="Val. loss")
            axs[1].plot(range(1, total_steps + 1), learning_rates)
        else:
            plt.figure(figsize=(16,9))
            plt.plot(range(1, total_steps + 1), train_losses, label="Train loss")
            plt.plot(range(1, total_steps + 2, steps_per_epoch), [train_losses[0], *val_losses], label="Val. loss")    
        save_plot("losses")
        plt.legend()
        plt.show()
    else:
        print("Finding learning rate ...")
        last_lr = LEARNING_RATE
        model = model.train()
        train_iter = iter(dataloader_train)
        count = 0
        while len(train_losses) < 10 or (last_lr < 2.0 and train_losses[0]*1.3 > train_losses[-1]):
            try:
                images, classes = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader_train)
                images, classes = next(train_iter)

            images = images.to(device)
            classes = classes.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = F.cross_entropy(output, classes)
            accelerator.backward(loss)
            optimizer.step()

            last_lr = lr_scheduler.get_last_lr()[0]
            learning_rates.append(last_lr)
            lr_scheduler.step()

            train_loss = loss.detach().cpu().item()

            train_losses.append(train_loss)
            count += 1

        train_losses = train_losses[:-1]
        learning_rates = learning_rates[:-1]
        max_idx = len(train_losses) - np.argmin(train_losses[::-1]) - 1
        max_lr = learning_rates[max_idx]
        middle = max_lr / 10.0
        middle_idx = len(train_losses) - np.argmin(np.abs(np.array(learning_rates) - middle)[::-1]) - 1
        middle_tested = learning_rates[middle_idx]
        print(f"Max. lr = {max_lr},  middle (not really) = {middle} ({middle_tested})")

        plt.figure(figsize=(16,9))
        plt.plot(learning_rates, train_losses)
        plt.scatter(max_lr, train_losses[max_idx], marker='.', color="red", s=300)
        plt.scatter(middle_tested, train_losses[middle_idx], marker='.', color="green", s=300)
        plt.xscale("log")
        save_plot("lr_finder")
        plt.show()
