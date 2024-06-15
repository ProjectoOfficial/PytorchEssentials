###############################################################################
# 
# Lecture 3 on Pytorch: Broadcasting, Squeeze, Unsqueeze, Indexing, and Slicing
#
# This script shows how to perform broadcasting, squeeze, unsqueeze, indexing, 
# and slicing in PyTorch
#
# Link: https://www.youtube.com/watch?v=zmT8tg71ak8
# Link: https://projectoofficial.github.io/
#
# @author: Daniel - PhD Candidate in ICT @ AImageLab
# 
###############################################################################

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# since we will use random tensors, it is better to fix the seed
torch.manual_seed(42)


################### BROADCASTING
# Here broadcasting is automatically done. What happen under the hood is that the
# smaller tensor is expanded to match the shape of the larger tensor.
def broadcasting():
    z = torch.ones([2, 2], dtype=torch.float32, device=device)
    y = torch.rand([2, 1], dtype=torch.float32, device=device)

    print(z * 3)

    # The previous is equivalent to the following
    print(z * torch.tensor([3, 3], dtype=torch.float32, device=device))

    # Broadcast takes place also with tensors of different shapes
    print(z * y)

    # The previous is equivalent to the following
    print(z * y.expand(z.shape))

    # We can even drop the last dimension of y that it will be still automatically broadcasted
    print(y.squeeze().shape)
    print(z * y.squeeze())

    # it works even if z is multi-dimensional, as long as its dimensions are multiples of y's dimensions
    z = torch.ones([2, 2, 4, 2], dtype=torch.float32, device=device)
    y = torch.ones([2], dtype=torch.float32, device=device) * 2
    print(z * y.squeeze())

    # it does not work if the dimensions are not multiples
    try:
        z = torch.ones([2, 2, 4, 2], dtype=torch.float32, device=device)
        y = torch.ones([3], dtype=torch.float32, device=device) * 2
        print(z * y.squeeze())
    except RuntimeError as e:
        print(e)

################### SQUEEZE AND UNSQUEEZE
# Squeeze removes all the dimensions of size 1
def squeeze_unsqueeze():
    z = torch.ones([2, 1, 2, 1], dtype=torch.float32, device=device)
    print(z.squeeze().shape)

    # squeeze can take a dimension as argument
    print(z.squeeze(1).shape)

    # Unsqueeze adds a dimension of size 1
    z = torch.ones([2, 2], dtype=torch.float32, device=device)
    print(z.unsqueeze(0).shape)

################### INDEXING AND SLICING
def indexing_slicing():
    # Indexing and slicing works as in numpy
    z = torch.ones([10, 2, 3], dtype=torch.float32, device=device)

    # get the first row
    print(z[:, 0])
    # get the first column
    print(z[0, :])
    # get the last dimension
    print(z[..., -1])

    # we can also use boolean masks
    z = torch.tensor([-1, 9, 3, -34, 12], dtype=torch.float32, device=device)
    mask = z > 0
    print(z[mask])

    # we can also use the where function: where(condition, x, y)
    # torch.where returns x if condition is True, y otherwise
    print(torch.where(mask, z, torch.zeros_like(z)))

    # we can also use the gather function
    # gather(input, dim, index)
    # input: tensor from which to gather values
    # dim: the dimension along which to index
    # index: the indices of the values to gather
    z = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32, device=device)
    print(torch.gather(z, 1, torch.tensor([[0], [1], [0]], device=device)))
    

if __name__ == "__main__":
    # broadcasting()
    # squeeze_unsqueeze()
    indexing_slicing()