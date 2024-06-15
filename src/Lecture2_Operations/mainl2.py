###############################################################################
# 
# Lecture 2 on Pytorch: Operations
#
# This script shows how to perform tensor operations in PyTorch 
#
# Link: https://www.youtube.com/watch?v=Vf2rSP1Ki40
# Link: https://projectoofficial.github.io/
#
# @author: Daniel - PhD Candidate in ICT @ AImageLab
# 
###############################################################################

import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Stride is a property of the tensor which indicates the number of elements in the memory 
# between two consecutive elements in the tensor dimension
z = torch.zeros((2, 2), dtype=torch.float64, device=device)
print(z.stride())

# pytorch operations allow us to perform element-wise operations on tensors
# For instance, we can multiply a tensor with a scalar
z = torch.ones((2, 2), dtype=torch.float64, device=device) * 2
print(z)

# We can also multiply two tensors element-wise, but they must have the same shape
y = torch.ones((2, 2), dtype=torch.float64, device=device) * 6
k = z * y
print(k)

# We can perform element-wise addition, subtraction, division, and exponentiation as well
k = z + y
print(k) # Addition

k = z - y
print(k) # Subtraction

k = z / y
print(k) # Division

k = z ** y
print(k) # Exponentiation

# We can perform these operations in some cases where the tensors have different shapes
# as long as the shapes are broadcastable
# The tensor must have the same shape except for the one dimension
y = torch.ones((1, 2), dtype=torch.float64, device=device) * 6
k = z - y
print(k)

# Operations such as sum, mean, max, min, etc. can be performed on tensors
# These operations can be performed along a specific dimension
# This dimension will obviously collapse in one resulting element
# For instance, we can sum all the elements of a tensor
z = torch.ones((2, 4), dtype=torch.float64, device=device) * 2
print(z)

k = z.sum(dim=1)
print(k)
print(k.shape)

# It is possible to preserve the dimension of the resulting tensor by setting keepdim=True
k = z.sum(dim=1, keepdim=True)
print(k)
print(k.shape)

# To concatenate two tensors along a specific dimension, we can use torch.cat
# The tensors must have the same shape except for the dimension along which they are concatenated
z = torch.ones((2, 2), dtype=torch.float64, device=device) * 2
y = torch.ones((2, 2), dtype=torch.float64, device=device) * 6
k = torch.cat((z, y), dim=0)
print(k)
print(k.shape)

# For further information, check the official documentation