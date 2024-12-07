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
