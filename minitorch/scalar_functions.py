from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the scalar function to the given values.

        Args:
        ----
            *vals (ScalarLike): Variable number of scalar-like inputs.

        Returns:
        -------
            Scalar: A new Scalar object with the result of the function application.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the addition function.

        Args:
        ----
            ctx (Context): The context object (unused in this case).
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of a + b.

        """
        return float(operators.add(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of the addition function.

        Args:
        ----
            ctx (Context): The context object (unused in this case).
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to inputs a and b.

        """
        return float(d_output), float(d_output)


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the natural logarithm function.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The natural logarithm of a.

        """
        ctx.save_for_backward(a)
        return float(operators.log(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the natural logarithm function.

        Args:
        ----
            ctx (Context): The context object containing saved values from forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return float(operators.log_back(a, d_output))


# TODO: Implement for Task 1.2.


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: The result of a * b.

        """
        ctx.save_for_backward(a, b)
        return float(operators.mul(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The context object containing saved values from forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to inputs a and b.

        """
        a, b = ctx.saved_values
        return float(b * d_output), float(a * d_output)


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the inverse function.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The result of 1/a.

        """
        ctx.save_for_backward(a)
        return float(operators.inv(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the inverse function.

        Args:
        ----
            ctx (Context): The context object containing saved values from forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return float(operators.inv_back(a, d_output))


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the negation function.

        Args:
        ----
            ctx (Context): The context object (unused in this case).
            a (float): The input value.

        Returns:
        -------
            float: The negation of a.

        """
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the negation function.

        Args:
        ----
            ctx (Context): The context object (unused in this case).
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        return float(-d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the sigmoid function.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The sigmoid of a.

        """
        sigmoid_value = float(operators.sigmoid(a))
        ctx.save_for_backward(sigmoid_value)
        return float(sigmoid_value)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the sigmoid function.

        Args:
        ----
            ctx (Context): The context object containing saved values from forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (sigmoid_value,) = ctx.saved_values
        return float(sigmoid_value * (1 - sigmoid_value) * d_output)


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The ReLU of a.

        """
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context object containing saved values from forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return float(operators.relu_back(a, d_output))


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the exponential function.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The exponential of a.

        """
        exp_value = float(operators.exp(a))
        ctx.save_for_backward(exp_value)
        return float(exp_value)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the exponential function.

        Args:
        ----
            ctx (Context): The context object containing saved values from forward pass.
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (exp_value,) = ctx.saved_values
        return float(exp_value * d_output)


class LT(ScalarFunction):
    """Less-than function $f(x) = 1.0 if x < y else 0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the less-than function.

        Args:
        ----
            ctx (Context): The context object (unused in this case).
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: 1.0 if a < b, else 0.0.

        """
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass of the less-than function.

        Args:
        ----
            ctx (Context): The context object (unused in this case).
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to inputs a and b (always 0.0).

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x) = 1.0 if x == y else 0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the equality function.

        Args:
        ----
            ctx (Context): The context object (unused in this case).
            a (float): The first input.
            b (float): The second input.

        Returns:
        -------
            float: 1.0 if a == b, else 0.0.

        """
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass of the equality function.

        Args:
        ----
            ctx (Context): The context object (unused in this case).
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to inputs a and b (always 0.0).

        """
        return 0.0, 0.0
