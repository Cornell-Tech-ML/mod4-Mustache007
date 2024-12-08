from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off

def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # TODO: Implement for Task 4.3.
    pooled_height = height // kh
    pooled_width = width // kw

    # Reshape input into tiles
    tiled_tensor = input.contiguous()
    tiled_tensor = tiled_tensor.view(batch, channel, pooled_height, kh, pooled_width, kw)
    tiled_tensor = tiled_tensor.permute(0, 1, 2, 4, 3, 5)
    tiled_tensor = tiled_tensor.contiguous()
    tiled_tensor = tiled_tensor.view(batch, channel, pooled_height, pooled_width, kh * kw)

    return tiled_tensor, pooled_height, pooled_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D."""
    # Use tile to reshape input for pooling
    tiled_input, pooled_height, pooled_width = tile(input, kernel)
    
    # Take mean over last dimension (the tiles)
    pooled_output = tiled_input.mean(4)
    pooled_output = pooled_output.contiguous()
    
    # Ensure the output has the correct shape (batch, channel, pooled_height, pooled_width)
    return pooled_output.contiguous().view(pooled_output.shape[0], pooled_output.shape[1], pooled_height, pooled_width)



reduce_max = FastOps.reduce(operators.max, -float("inf"))

def argmax(input_tensor: Tensor, dimension: int) -> Tensor:
    """Returns a one-hot tensor with 1s at the argmax positions along the specified dimension."""
    max_values = reduce_max(input_tensor, dimension)
    return max_values == input_tensor


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, dimension: Tensor) -> Tensor:
        """Computes the forward pass of the max operation.
        
        Args:
        ----
            ctx (Context): The context for the operation.
            input_tensor (Tensor): The input tensor.
            dimension (int): The dimension along which to compute the maximum.
            
        Returns:
            Tensor: The result of the tensor operation.

        """
        ctx.save_for_backward(input_tensor, dimension)
        return reduce_max(input_tensor, int(dimension.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the backward pass for the tensor operation.
        
        Args:
            ctx (Context): The context object containing information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor.
            
        Returns:
        -------
            Tensor: The gradient of the input tensor.

        """
        input_tensor, dimension = ctx.saved_values
        return argmax(input_tensor, int(dimension.item())) * grad_output, 0.0


def max(input_tensor: Tensor, dimension: int) -> Tensor:
    """Returns max values along dimension."""
    return Max.apply(input_tensor, tensor(dimension))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Apply softmax along specified dimension."""
    max_values = max(input, dim)
    shifted_input = input - max_values
    exp_values = shifted_input.exp()
    return exp_values / exp_values.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute log softmax along dimension."""
    max_values = max(input, dim)
    shifted_input = input - max_values
    return input - (max_values + shifted_input.exp().sum(dim=dim).log())


def maxpool2d(input_tensor: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D."""
    # Use tile to reshape input for pooling
    tiled_tensor, pooled_height, pooled_width = tile(input_tensor, kernel)
    
    # Take max over last dimension (the tiles)
    pooled_output = max(tiled_tensor, dimension=4)
    pooled_output = pooled_output.contiguous()
    
    # Ensure the output has the correct shape (batch, channel, pooled_height, pooled_width)
    return pooled_output.contiguous().view(pooled_output.shape[0], pooled_output.shape[1], pooled_height, pooled_width)


def dropout(input: Tensor, drop_prob: float, ignore: bool = False) -> Tensor:
    """Randomly zero elements with probability drop_prob and scale remaining elements by 1/(1-drop_prob)."""
    if ignore or drop_prob <= 0.0:
        return input
    
    if drop_prob >= 1.0:
        return input * 0.0
    
    dropout_mask = rand(input.shape) > drop_prob
    return input * dropout_mask