###############################################################################
# 
# Lecture 7 on Pytorch: Convolutions and Blocks
#
# This script shows how to create a Convolutional Neural Network (CNN) with
# Depthwise Separable Convolutions and Squeeze-and-Excitation blocks.
# We also see how to augment the data in order to improve the generalization.
# Training is performed on the CIFAR-10 dataset.
#
# Link: https://www.youtube.com/watch?v=vMGmyZAYR24
# Link: https://projectoofficial.github.io/
#
# @author: Daniel - PhD Candidate in ICT @ AImageLab
# 
###############################################################################


import argparse

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T

from tqdm import tqdm

def collate_fn(batch: tuple, device: torch.device):
    images, labels = zip(*batch)
    images = torch.stack(images).to(device)
    labels = torch.tensor(labels).to(device)

    return images, labels

def get_dataset(batch_size: int, num_workers: int, device: torch.device):
    data_path = '/tmp/data'

    train_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.RandomErasing()
    ])

    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transforms)
    val_set = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch, device))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch, device))

    return train_loader, val_loader

class ChannelSELayer(nn.Module):
    def __init__(self, in_channels: int, reduction: int):
        super(ChannelSELayer, self).__init__()

        hidden_channels = in_channels // reduction
        self.reduction_ratio = reduction
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.fc2 = nn.Linear(hidden_channels, in_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, H, W = x.size()
        # Average along each channel
        squeeze_tensor = x.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(x, fc_out_2.view(a, b, 1, 1))
        return output_tensor
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, reduction: int):
        super(ConvBlock, self).__init__()

        self.expander = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1)
        self.dwconv = nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size, stride, padding, groups=out_channels * 4)
        self.bn = nn.BatchNorm2d(out_channels * 4)
        self.se = ChannelSELayer(out_channels * 4, reduction)
        self.reductor = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, stride=1)

        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_connection(x)

        x = self.expander(x)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.se(x)
        x = self.reductor(x) + skip
        return x
    
class ConvNet(nn.Module):
    def __init__(self, in_channels: int, out_classes: int, reduction: int):
        super(ConvNet, self).__init__()

        self.arch = nn.Sequential(
            ConvBlock(in_channels, out_channels=96, kernel_size=3, stride=1, padding=1, reduction=reduction),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1, reduction=reduction),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, reduction=reduction),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(in_channels=384, out_channels=738, kernel_size=3, stride=1, padding=1, reduction=reduction),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(738, out_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.arch(x)
        x = self.classifier(x)
        return x

def ckpts_manager(ckpt_path: str, model: nn.Module, optimizer: torch.optim.Optimizer, mode: str):
    if ckpt_path is None or ckpt_path == '':
        print("No checkpoint path provided!")
        return model, optimizer

    if mode == 'load':
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    elif mode == 'save':
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(ckpt, ckpt_path)

    return model, optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch-size", type=int, default=512, help="size of the batch of images")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("-r", "--reduction", type=int, default=4, help="reduction ratio for SE block")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4, help="learning rate for the optimizer")
    parser.add_argument("-nw", "--num-workers", type=int, default=0, help="number of workers for the dataloader")
    parser.add_argument('-sw', '--save-weights', type=str, default='weights.pth', help='path to save the weights')
    parser.add_argument('-lw', '--load-weights', type=str, default=None, help='path to load the weights')
    parser.add_argument('-cp', '--checkpoint', type=str, default=None, help='path to a checkpoint to load or store')
    

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataset(args.batch_size, num_workers=args.num_workers, device=device)

    model = ConvNet(in_channels=3, out_classes=10, reduction=args.reduction).to(device)
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.load_weights:
        model, optimizer = ckpts_manager(args.load_weights, model, optimizer, mode='load')
        print("Weights loaded!")

    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        pbar.set_description(f"Epoch {epoch + 1} | Loss: {loss.item():.4f} | Accuracy: {100 * correct / total:.2f}%")
        pbar.update(1)

    print("Training completed!")
    
    if args.checkpoint:
        model, optimizer = ckpts_manager(args.checkpoint, model, optimizer, mode='save')
        print("Checkpoint saved!")
    
    exit(0)

        