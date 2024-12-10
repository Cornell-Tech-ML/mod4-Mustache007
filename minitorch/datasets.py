import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate a list of N random 2D points, where each point is represented as a tuple of two random float values between 0 and 1.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        List: A list of N randomly generated points

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """A dataclass representing a graph of 2D points and their associated labels."""

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a simple dataset where labels are determined by the x_1 coordinate.

    Args:
    ----
        N: The number of points.

    Returns:
    -------
        Graph: A graph with points and labels (1 if x_1 < 0.5, else 0).

    """
    X = make_pts(N)
    y = [1 if x_1 < 0.5 else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a dataset where labels are determined by the sum of x_1 and x_2.

    Args:
    ----
        N: The number of points.

    Returns:
    -------
        Graph: A graph with points and labels (1 if x_1 + x_2 < 0.5, else 0).

    """
    X = make_pts(N)
    y = [1 if x_1 + x_2 < 0.5 else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a dataset where labels are determined by x_1 falling outside the range [0.2, 0.8].

    Args:
    ----
        N: The number of points.

    Returns:
    -------
        Graph: A graph with points and labels (1 if x_1 < 0.2 or x_1 > 0.8, else 0).

    """
    X = make_pts(N)
    y = [1 if x_1 < 0.2 or x_1 > 0.8 else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates a dataset where labels are determined by the XOR operation on x_1 and x_2.

    Args:
    ----
        N: The number of points.

    Returns:
    -------
        Graph: A graph with points and labels (1 if XOR condition is met, else 0).

    """
    X = make_pts(N)
    y = [
        1 if (x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5) else 0
        for x_1, x_2 in X
    ]
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a dataset where labels are determined by the distance from the center.

    Args:
    ----
        N: The number of points.

    Returns:
    -------
        Graph: A graph with points and labels (1 if the point is outside the radius 0.1, else 0).

    """
    X = make_pts(N)
    y = [1 if (x_1 - 0.5) ** 2 + (x_2 - 0.5) ** 2 > 0.1 else 0 for x_1, x_2 in X]
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a spiral dataset.

    Args:
    ----
        N: The number of points.

    Returns:
    -------
        Graph: A graph with spiral points and corresponding labels.

    """

    def x(t: float) -> float:
        """Calculate the x-coordinate of a point on the spiral.

        Args:
        ----
            t: The parameter value along the spiral.

        Returns:
        -------
            float: The x-coordinate of the point on the spiral.

        """
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        """Calculate the y-coordinate of a point on the spiral.

        Args:
        ----
            t: The parameter value along the spiral.

        Returns:
        -------
            float: The y-coordinate of the point on the spiral.

        """
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X += [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
