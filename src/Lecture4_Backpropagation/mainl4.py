import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# EPOCHS: number of times the entire dataset is passed through the network
EPOCHS = 500

# N: batch size, input_dimension: input data dimension, hidden_dimension: hidden layer dimension, output_dimension: output data dimension
N, input_dimension, hidden_dimension, output_dimension = 64, 1000, 100, 10
x = torch.randn(N, input_dimension, device=device) # dataset made of random numbers
y = torch.randn(N, output_dimension, device=device) # dataset's label made of random numbers 

w1 = torch.randn(input_dimension, hidden_dimension, device=device, requires_grad=True) # input weight matrix
w2 = torch.randn(hidden_dimension, output_dimension, device=device, requires_grad=True) # output weight matrix

learning_rate = 1e-6

for epoch in range(EPOCHS):
    # this is not necessary, but I want to make it clear that the input data is x
    input_data = x

    # first the input data is multiplied by the input weight matrix through matrix multiplication
    # then the data is activated by ReLU (if the value is less than 0, it is changed to 0, otherwise it remains the same)
    hidden_data = torch.matmul(input_data, w1)
    hidden_data_activated = hidden_data.clamp(min=0)
    
    # the activated data is multiplied by the output weight matrix through matrix multiplication
    output_data = torch.matmul(hidden_data_activated, w2)

    # the loss is calculated by taking the sum of the squared difference between the output data and the label
    # this is equivalent to the mean squared error
    loss = (output_data - y).pow(2).sum()

    # the gradient of the loss with respect to the input weight matrix and the output weight matrix is calculated
    loss.backward()
    print(loss.item())

    with torch.no_grad():
        # the input weight matrix and the output weight matrix are updated by subtracting the product of the learning rate and the gradient
        # the gradient represents the direction in which the loss decreases, the learning rate represents the size of the step
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # the gradient is reset to 0
        w1.grad.zero_()
        w2.grad.zero_()