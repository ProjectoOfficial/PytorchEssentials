###############################################################################
# 
# Lecture 1 on Pytorch: Tensors
#
# This script shows how to create tensors in PyTorch, how to inspect them,
# and how to move them to a specific device, with a specific dtype.
#
# Link: https://www.youtube.com/watch?v=vMGmyZAYR24
# Link: https://projectoofficial.github.io/
#
# @author: Daniel - PhD Candidate in ICT @ AImageLab
# 
###############################################################################

import torch

# Check if CUDA is correctly installed and the GPU is available
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"How many CUDA devices are available? {torch.cuda.device_count()}")
print(f"Name of the CUDA device: {torch.cuda.get_device_name(0)}")


# Select the device to be used for the computation
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create a tensor and send it to the device:
# 1. The tensor is directly created on the device (more efficient)
z = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 8]], dtype=torch.float32, device=device)

# 2. The tensor is created on the CPU and then moved to the device
x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 8]]).to(torch.float32).to(device)

print(z)
print(x)

# With z.shape we can get the shape of the tensor which indicates the number of elements in each dimension
print(z.shape)

# With z.size() we can get the total number of elements in the tensor
print(z.size())

# With z.numel() we can get the total number of elements in the tensor as well
print(z.numel())

# We can also get the data type of the tensor with z.dtype
print(z.dtype)

# Since we can also set the tensor's device, we can check the device of the tensor with z.device
print(z.device)

# Now, it is possible to create a tensor manually as we've seen before,
# but PyTorch provides a variety of functions to create tensors with specific properties.
# For instance, we can create a tensor with all zeros with torch.zeros(shape)
z = torch.zeros((4, 4), dtype=torch.float32, device=device)
print(z)

# Similarly, we can create a tensor with all ones with torch.ones(shape)
x = torch.ones((4, 4), dtype=torch.float32, device=device)
print(x)

# We can also create a tensor with random values with torch.rand(shape)
y = torch.rand((4, 4), device=device)
print(y)

# We can generate a tensor with random values from a normal distribution with torch.randn(shape)
y = torch.randn((4, 4), device=device)
print(y)

# We can also choose one value used to populate a tensor with torch.full(shape, value)
z = torch.full((4, 4), 42, device=device)
print(z)

# There are multiple ways to create tensors with specific properties, my suggestion is to check the documentation