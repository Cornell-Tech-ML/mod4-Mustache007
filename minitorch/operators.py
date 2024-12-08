"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        x: The first float number.
        y: The second float number.

    Returns:
    -------
        The float product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        x: The input value.

    Returns:
    -------
        The same value as input.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.

    Args:
    ----
        x: The first float number.
        y: The second float number.

    Returns:
    -------
        The float sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.

    Args:
    ----
        x: The number to negate.

    Returns:
    -------
        The negated value of x.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        1.0 if x is less than y, 0.0 otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        1.0 if x and y are equal, 0.0 otherwise.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
        The larger of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close in value.

    Args:
    ----
        x: The first number.
        y: The second number.

    Returns:
    -------
       | x - y | < 1e-2

    """
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        x: The input value.

    Returns:
    -------
        The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function.

    Args:
    ----
        x: The input value.

    Returns:
    -------
        The result of applying ReLU to x.

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm.

    Args:
    ----
        x: The input value.
        EPS: A small number to avoid log(0) being infinite.

    Returns:
    -------
        The natural logarithm of x.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
        x: The input value.

    Returns:
    -------
        The exponential of x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal.

    Args:
    ----
        x: The input value.

    Returns:
    -------
        The reciprocal of x.

    """
    return 1.0 / x


def log_back(x: float, a: float) -> float:
    """Computes the derivative of the logarithm times a second argument.

    Args:
    ----
        x: The input value for the logarithm.
        a: The multiplier for the derivative.

    Returns:
    -------
        The derivative of the logarithm of x times a.

    """
    return a / (x + EPS)


def inv_back(x: float, a: float) -> float:
    """Computes the derivative of the reciprocal times a second argument.

    Args:
    ----
        x: The input value for the reciprocal.
        a: The multiplier for the derivative.

    Returns:
    -------
        The derivative of the reciprocal of x times a.

    """
    return -(1.0 / (x * x)) * a


def relu_back(x: float, a: float) -> float:
    """Computes the derivative of ReLU times a second argument.

    Args:
    ----
        x: The input value for ReLU.
        a: The multiplier for the derivative.

    Returns:
    -------
        The derivative of ReLU applied to x times a.

    """
    return a if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map function.

    Args:
    ----
        fn (Callable[[float], float]): A function that takes a float as input and returns a float.

    Returns:
    -------
        A function that takes a list, applies fn to each element, and returns a new list.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith function.

    Args:
    ----
        fn: combine two floats

    Returns:
    -------
        Generate a new list by applying fn to the corresponding elements of the two input lists.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce function.

    Args:
    ----
        fn (Callable[[float, float], float]): A function that takes two floats as input and returns a float.
        start (float): The initial value for the reduction.

    Returns:
    -------
        A function that takes an iterable of floats, applies the reduction using fn and start, and returns a float.

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(list: Iterable[float]) -> Iterable[float]:
    """Negates all elements in a list using map function.

    Args:
    ----
        list: input list of float values.

    Returns:
    -------
        An iterable containing the negated values of the input list.

    """
    return map(neg)(list)


def addLists(list1: Iterable[float], list2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith function.

    Args:
    ----
        list1: The first input list of float values.
        list2: The second input list of float values.

    Returns:
    -------
        An iterable containing the sum of corresponding elements from list1 and list2.

    """
    return zipWith(add)(list1, list2)


def sum(list: Iterable[float]) -> float:
    """Sum all elements in a list using reduce function.

    Args:
    ----
        list: The input list of float values to be summed.

    Returns:
    -------
        The sum of every elements in the input list.

    """
    return reduce(add, 0.0)(list)


def prod(list: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce function.

    Args:
    ----
        list: The input list of float values to be multiplied.

    Returns:
    -------
        The product of all elements in the input list.

    """
    return reduce(mul, 1.0)(list)
