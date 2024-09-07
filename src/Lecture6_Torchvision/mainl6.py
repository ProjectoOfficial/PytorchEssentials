###############################################################################
# 
# Lecture 6 on Pytorch: Torchvision
#
# This script shows how to create a simple linear network to classify the MNIST dataset.
# We use the torchvision library to download and load the dataset, and to apply transformations.
# We also use the DataLoader to create an iterable object that provides the data in batches.
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

# we use torchvision to work with image datasets
# we can download and load data, while also apply transforms on it
from torchvision.datasets import MNIST
import torchvision.transforms as T

from tqdm import tqdm
import logging

# Let's define a basic Linear network with 1024 as hidden dimension
# We use batch normalization, which normalizes tensors along the batch dimension
# to help the model to better generalize
class LinearNet(nn.Module):
    def __init__(self, in_channels: int, out_classes: int) -> None:
        super(LinearNet, self).__init__()

        self.arch = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, out_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.arch(x)

# a collate function is a special function executed before the dataloader provides a batch.
# it is very useful to apply further custom operation on the data before using it (e.g.
# you may add here positional encoding)
def collate_fn(batch: tuple, device: torch.device):
    images, labels = zip(*batch)
    images = torch.stack(images).to(device)
    labels = torch.tensor(labels).to(device)

    return images, labels

if __name__ == "__main__":
    # since we do not want to be bad programmer, we always need to make clear which parameters
    # the user can modify (useful for us to train multiple configurations)
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch-size", type=int, default=512, help="size of the batch of images")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="number of training epochs")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    logging.basicConfig(filename="Lecture6_Torchvision/training.log", level=logging.INFO)

    # A transform is always applied on data. Here first we transform an input image to tensor
    # since we will work then with tensors; then we normalize this tensor to lay in [-1, 1]
    # inteval thanks to 0.5 mean and 0.5 variance normalization (this helps the model to 
    # better generalize); finally we want to apply a custom transformation, we want to reshape
    # the tensor in order to make it linear (otherwise it does not fit into nn.Linear)

    # transform compose takes a list where order matters!
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5), (0.5)),
        T.Lambda(lambda x: x.view(-1))
    ])

    # here we define the dataset: 
    # - first param: specifies the path to the dataset folder within the filesystem
    # - second param: datasets are tipically split into (trainset, valset, testset)
    #                 thus we need to specify which split we want
    # - third param: the transform we wrote before
    trainset = MNIST("/tmp/data", train=True, download=True, transform=transform) # (50.000 images)
    testset = MNIST("/tmp/data", train=False, download=True, transform=transform) # (10.000 images)

    # The dataloader is an iterable object that we will use to take the current batch during training or testing
    # - first param: the set object --> trainset or testset in this case
    # - second param: in training it is better to shuffle data because otherwise the network may learn to classify
    #                 only by remembering the order of the input data
    # - third param: num workers are the number of process which actively are involved in loading the data.
    #                0 means auto, N can go up to your processor number of threads (you may need to set multiprocessing)
    # - fourth param: collate fn we wrote before, where we can pass also the device

    trainloader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda batch: collate_fn(batch, device))
    testloader = DataLoader(testset, args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda batch: collate_fn(batch, device))

    model = LinearNet(in_channels=784, out_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   
    numParameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {numParameters} parameters")
    logging.info(model)


    print("Training started!")
    pbar = tqdm(total=args.epochs, desc=f"EPOCH: 0 - running ...")
    for e in range(args.epochs):
        avg_loss = 0

        # Training Step: the output of a XXXXloader is always a tuple
        for (images, labels) in trainloader:
            predictions = model(images)

            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

        avg_loss /= len(trainloader) / args.batch_size

        # Validation Step
        correct = 0
        total = 0
        with torch.no_grad():
            for (images, labels) in testloader:
                predictions = model(images)

                # we take the max values --> the highes probabilities (in the model's opinion)
                _, predicted = torch.max(predictions, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        message = f"EPOCH: {e}: average loss is {avg_loss}, while accuracy is {accuracy}"
        pbar.set_description(message)
        logging.info(message)
        pbar.update(1)

