# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Wrapper around numba cuda jit decorator that enables device compilation."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Any, **kwargs: Any) -> FakeCUDAKernel:
    """Wrapper around numba cuda jit decorator."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """CUDA version of tensor elementwise zip."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """CUDA version of tensor reduce."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """CUDA version of matrix multiplication.

        Args:
        ----
            a (Tensor): Left tensor for multiplication
            b (Tensor): Right tensor for multiplication

        Returns:
        -------
            Tensor: Result of matrix multiplication with shape [..., a.shape[-2], b.shape[-1]]
            where ... represents broadcasted batch dimensions

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_position = index_to_position(in_index, in_strides)
            out[i] = fn(in_storage[in_position])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i < out_size:
            # Convert output index to indices
            to_index(i, out_shape, out_index)
            # Broadcast indices for both inputs
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # Get storage positions
            a_position = index_to_position(a_index, a_strides)
            b_position = index_to_position(b_index, b_strides)
            # Apply function to values at those positions
            out[i] = fn(a_storage[a_position], b_storage[b_position])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """CUDA sum practice kernel to prepare for reduce.

    Given an array of length n and out of size n // blockDIM,
    sums up each blockDim values into an out cell.

    Example:
    -------
        Input: [a1, a2, ..., a100]
        Output: [a1 + ... + a31, a32 + ... + a64, ...]

    Note: Each block must do the sum using shared memory.

    Args:
    ----
        out (Storage): Storage for output tensor
        a (Storage): Storage for input tensor
        size (int): Length of input tensor

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Load data into shared memory
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0

    cuda.syncthreads()
    # Parallel reduction in shared memory
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride and i + stride < size:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2
    # Write result for this block to global memory
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice CUDA sum implementation that uses shared memory.

    Args:
    ----
        a: Input tensor to sum

    Returns:
    -------
        TensorData containing partial sums computed by each thread block

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if out_pos < out_size:
            # Initialize shared memory with initial value
            shared_cache = cache
            shared_cache[pos] = reduce_value
            # Calculate starting position
            to_index(out_pos, out_shape, out_index)
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
            current_pos = index_to_position(out_index, a_strides)

            # Update cache if within bounds
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                if pos == 0:
                    shared_cache[pos] = fn(shared_cache[pos], a_storage[current_pos])
                else:
                    shared_cache[pos] = a_storage[current_pos]
                cuda.syncthreads()
                # Reduction within block
                stride = 1
                while stride < BLOCK_DIM:
                    if (pos & ((2 * stride) - 1)) == 0 and pos + stride < BLOCK_DIM:
                        shared_cache[pos] = fn(
                            shared_cache[pos], shared_cache[pos + stride]
                        )
                        cuda.syncthreads()
                    stride *= 2

            # Write final result
            if pos == 0:
                out[out_pos] = shared_cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    r"""Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    # Define block dimension for shared memory allocation
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.4.
    # Allocate shared memory for matrix tiles
    # Each block will work on BLOCK_DIM x BLOCK_DIM tiles of the input matrices
    matrix1_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    matrix2_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # Calculate global row and column indices for this thread
    # blockIdx: which block this thread belongs to
    # blockDim: number of threads per block
    # threadIdx: position of thread within its block
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # Get local thread indices within the block
    # These are used to access shared memory
    thread_row = cuda.threadIdx.x
    thread_col = cuda.threadIdx.y
    # Calculate linear index into global memory
    # For matrices stored in row-major order
    index = row * size + col
    # Only compute if within matrix bounds
    if row < size and col < size:
        # Load data from global to shared memory
        # Each thread loads one element of each matrix
        matrix1_cache[thread_row, thread_col] = a[index]
        matrix2_cache[thread_row, thread_col] = b[index]
        # Ensure all threads have loaded their data before proceeding
        cuda.syncthreads()
        # Compute dot product for this thread's output element
        result = 0
        for k in range(size):
            # Multiply corresponding elements and accumulate
            result += matrix1_cache[thread_row, k] * matrix2_cache[k, thread_col]
        # Write final result back to global memory
        out[index] = result


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice matrix multiplication implementation using CUDA.

    Args:
    ----
        a (Tensor): First input tensor (square matrix)
        b (Tensor): Second input tensor (square matrix)

    Returns:
    -------
        TensorData: Result of matrix multiplication

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y
    # TODO: Implement for Task 3.4.
    # Initialize accumulator for dot product result
    # Each thread maintains its own accumulator for its assigned output position
    acc = 0.0

    # Process the matrices in BLOCK_DIM x BLOCK_DIM tiles
    # This loop moves across the shared dimension (columns of A, rows of B)
    for block_start in range(0, a_shape[-1], BLOCK_DIM):
        # Initialize shared memory blocks to zero
        # Each thread zeros out its assigned position in both shared arrays
        a_shared[pi, pj] = 0
        b_shared[pi, pj] = 0
        # Ensure all threads have completed initialization before proceeding
        cuda.syncthreads()

        # Load a tile from matrix A into shared memory
        # Only threads within matrix bounds perform the load
        # Calculate global index using batch stride and matrix strides
        if i < a_shape[-2] and block_start + pj < a_shape[-1]:
            a_idx = (
                batch * a_batch_stride
                + i * a_strides[-2]
                + (block_start + pj) * a_strides[-1]
            )
            a_shared[pi, pj] = a_storage[a_idx]

        # Load a tile from matrix B into shared memory
        # Only threads within matrix bounds perform the load
        # Calculate global index using batch stride and matrix strides
        if block_start + pi < b_shape[-2] and j < b_shape[-1]:
            b_idx = (
                batch * b_batch_stride
                + (block_start + pi) * b_strides[-2]
                + j * b_strides[-1]
            )
            b_shared[pi, pj] = b_storage[b_idx]

        # Ensure all threads have completed loading data before computation
        cuda.syncthreads()

        # Compute partial dot product for this tile
        # Only threads that will produce valid output elements participate
        if i < out_shape[-2] and j < out_shape[-1]:
            # Multiply and accumulate across the current tile
            # k iterates over the shared dimension within the current tile
            for k in range(min(BLOCK_DIM, a_shape[-1] - block_start)):
                acc += a_shared[pi, k] * b_shared[k, pj]

        # Ensure all threads complete computation before loading next tile
        cuda.syncthreads()

    # After processing all tiles, write final accumulated result to global memory
    # Only threads within output matrix bounds write their result
    if i < out_shape[-2] and j < out_shape[-1]:
        # Calculate global output index using batch and matrix strides
        out_idx = batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
        out[out_idx] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
