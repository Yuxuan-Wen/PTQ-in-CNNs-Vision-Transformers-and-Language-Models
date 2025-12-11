"""Train CIFAR10 with PyTorch."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import quantization
import os
import argparse

from models import *

# Added by me
from models.resnet20 import ResNet20, ResBlock
from utils import (
    progress_bar,
    inplace_quantize_layers,
    enable_calibrate,
    disable_calibrate,
    calibrate_adaround,
    add_module_dict,
)

# Added for saving unique checkpoints to avoid overwriting
import datetime

# Modified to include more options required for our project
parser = argparse.ArgumentParser(
    description="ResNet Training. Select a model from this list: \nResNet18, \nResNet20, \nResNet34, \nResNet50"
)
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--type", choices=["fp32", "PTQ", "QAT"], help="choose train fp32, PTQ or QAT"
)
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--dorefa", "-d", action="store_true", help="use dorefa to quantizate"
)
parser.add_argument(
    "--Histogram", action="store_true", help="use HistogramObserver to quantizate"
)
parser.add_argument("--omse", action="store_true", help="use omse to quantizate")
parser.add_argument("--lsq", action="store_true", help="use lsq to quantizate")
parser.add_argument(
    "--bias_correction", action="store_true", help="use bias_correction to quantizate"
)
parser.add_argument(
    "--level", default="L", choices=["L", "C"], help="per_channel or per_tensor"
)
parser.add_argument("--path", default="./checkpoint/", help="model saved path")

parser.add_argument(
    "--adaround", action="store_true", help="use adaround to quantizate"
)
parser.add_argument("--adaround-iter", default=1000, type=int)
parser.add_argument(
    "--b_start",
    default=20,
    type=int,
    help="temperature at the beginning of calibration",
)
parser.add_argument(
    "--b_end", default=2, type=int, help="temperature at the end of calibration"
)
parser.add_argument(
    "--warmup",
    default=0.2,
    type=float,
    help="in the warmup period no regularization is applied",
)
parser.add_argument(
    "--checkpoint_save",
    default="ckpt.pth",
    type=str,
    help="checkpoint name to save the model",
)
parser.add_argument(
    "--checkpoint_file",
    default="ckpt.pth",
    type=str,
    help="checkpoint name to load the model",
)
parser.add_argument(
    "--model", default="resnet20", type=str, help="selected model for training"
)
parser.add_argument(
    "--quantization_saved",
    default="ckpt_q.pth",
    type=str,
    help="checkpoint name to save the quantized model",
)
# Adding to adjust bit-width as a part of our study
parser.add_argument(
    "--bit",
    default=8,
    type=int,
    help="bit-width for quantizaiton (default: 8 bits)",
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Typical for CIRFAR10 training
train_epochs = 180

# Data
print("==> Preparing data..")

# MNIST Dataset (replaced by CIFAR10) -> Uncomment if you want to use this dataset
# trainset = torchvision.datasets.MNIST(
#    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
# trainloader = torch.utils.data.DataLoader(
#    trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.MNIST(
#    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
# testloader = torch.utils.data.DataLoader(
#    testset, batch_size=100, shuffle=False, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#           'dog', 'frog', 'horse', 'ship', 'truck')

# Normalization values for CIFAR10
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# CIFAR10 Dataset (Augmentation strategy was copied from class HW)
trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            # Augmentation
            transforms.RandomCrop(32, padding=4),  # Random cropping with padding = 4
            transforms.RandomHorizontalFlip(),  # Random horizontal flipping
            # Required transformations
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    ),
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    ),
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

# CIFAR10 Classes (fixed from previous code version...)
classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Model
print("==> Building model..")

# Generate unique timestamp for checkpoint filename
now = datetime.datetime.now()
formatted_string = now.strftime("%Y%m%d_%H%M%S")

# Model selection (ResNet family)
if args.model == "resnet20":
    net = ResNet20(ResBlock, [3, 3, 3], num_classes=10)
    net = net.to(device)
elif args.model == "resnet18":
    net = torchvision.models.resnet18(weights=None, num_classes=10)
    net = net.to(device)
elif args.model == "resnet34":
    net = torchvision.models.resnet34(weights=None, num_classes=10)
    net = net.to(device)
elif args.model == "resnet50":
    net = torchvision.models.resnet50(weights=None, num_classes=10)
    net = net.to(device)
else:
    print(f"Error: Model '{args.model}' not supported")
    print("Supported models: resnet20, resnet18, resnet34, resnet50")
    exit()

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/" + args.checkpoint_file)
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

if args.type == "PTQ" or args.lsq:
    # Check if the file exists
    if not os.path.isfile("./checkpoint/" + args.checkpoint_file):
        print("Error: no checkpoint file found!")
        exit()
    # Load the checkpoint and state dictionary
    checkpoint = torch.load("./checkpoint/" + args.checkpoint_file)
    new_state_dict = add_module_dict(checkpoint["net"])
    # net.load_state_dict(checkpoint['net'])
    net.load_state_dict(new_state_dict)

if args.type == "PTQ" or args.type == "QAT":
    net = inplace_quantize_layers(
        net,
        len(trainloader) * train_epochs,
        ptq=True if args.type == "PTQ" else False,
        dorefa=args.dorefa,
        Histogram=args.Histogram,
        level=args.level,
        omse=args.omse,
        adaround=args.adaround,
        bias_correction=args.bias_correction,
        lsq=args.lsq,
        # Added to try different bit-widths
        bit=args.bit,
    )
    net = net.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[60, 120, 160], gamma=0.2
)


# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )


def calibrate():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx == 10:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(trainloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )


def calibrate_ada(net):
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx == 10:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )
    return net


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:

        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }

        # Make checkpoint directory if it doesn't exist
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")

        if args.type == "fp32":
            # If the file already exists, change the name to include timestamp
            # This is to avoid overwriting a previously saved checkpoint
            checkpoint_path = f"./checkpoint/{args.checkpoint_save}"

            torch.save(state, checkpoint_path)
            # Also save as the default name for easy resuming
            torch.save(state, "./checkpoint/" + args.checkpoint_save)
            print(f"Checkpoint saved to: {checkpoint_path}")
        else:
            # Save quantized model with user-defined name
            checkpoint_path = f"./checkpoint/{args.quantization_saved}"
            torch.save(state, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
        best_acc = acc


for epoch in range(start_epoch, start_epoch + train_epochs):
    if epoch == start_epoch:
        enable_calibrate(net)
        calibrate()
        disable_calibrate(net)
        if args.adaround:
            calibrate_adaround(
                net,
                args.adaround_iter,
                args.b_start,
                args.b_end,
                args.warmup,
                trainloader,
                device,
            )
        test(epoch)
        if args.type == "PTQ":
            break
    else:
        train(epoch)
        test(epoch)
        scheduler.step()
