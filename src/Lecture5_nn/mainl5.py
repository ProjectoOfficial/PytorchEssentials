###############################################################################
# 
# Lecture 5 on Pytorch: Neural Networks
#
# This script shows how to create a simple linear network to perform regression.
# We use the nn.Module class to define the network and the forward method to 
# define the forward pass. We also define a training loop to train the network.
#
# Link: https://www.youtube.com/watch?v=vMGmyZAYR24
# Link: https://projectoofficial.github.io/
#
# @author: Daniel - PhD Candidate in ICT @ AImageLab
# 
###############################################################################

import torch
from torch import nn
import torch.optim as optim

# A neural network is defined as a class that inherits from nn.Module
# The class has two main methods: __init__ and forward
# __init__ is used to define the layers and attributes of the network
# forward is used to define the forward pass of the network
class Network(nn.Module):
    def __init__ (self, input_dimension: int, hidden_dimension: int, output_dimension: int) -> None:
        super(Network, self).__init__()

        # nn.Sequential is a container for modules, modules are applied in the order they are passed
        self.net = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension), # input linear layer
            nn.ReLU(), # activation function
            nn.Linear(hidden_dimension, output_dimension) # output linear layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# set seed for random generated numbers to allow reproducibility
def set_seed(seed: int=42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # EPOCHS: number of times the entire dataset is passed through the network
    EPOCHS = 500

    # N: batch size, input_dimension: input data dimension, hidden_dimension: hidden layer dimension, output_dimension: output data dimension
    N, input_dimension, hidden_dimension, output_dimension = 64, 1000, 100, 10
    x = torch.randn(N, input_dimension, device=device) # dataset made of random numbers
    y = torch.randn(N, output_dimension, device=device) # dataset's label made of random numbers

    # the model is moved to the device
    model = Network(input_dimension, hidden_dimension, output_dimension)
    model = model.to(device)

    criterion = nn.MSELoss(reduction="sum") # mean squared error loss
    optimizer = optim.SGD(model.parameters(), lr=1e-4) # stochastic gradient descent optimizer 

    for epoch in range(EPOCHS):
        
        # the gradient is reset to 0 before running the model and calculating the loss
        optimizer.zero_grad()

        # the model is run with the input data
        # the loss is calculated by taking the sum of the squared difference between the output data and the label
        prediction = model(x)
        loss = criterion(prediction, y)

        loss.backward()
        print(loss.item())

        # the optimizer updates the model's parameters
        optimizer.step()
        
    print("\n")

    # VALIDATION STEP:
    # we define a validation dataset to test the goodness of our model
    # we first need to check if it does not overfit (it does...)
    x_val = torch.randn(N, input_dimension, device=device)
    y_val = torch.randn(N, output_dimension, device=device)

    # since we do not have to backpropagate because we do not want to train the model on the validation set
    # (otherwise AI would not make sense) we set torch.no_grad(). This allows the model not to retain gradients,
    # which means faster runtime and less memory footprint.
    with torch.no_grad():
        prediction = model(x_val)
        loss = criterion(prediction, y_val)

        print(f"validation loss: {loss.item()}")

