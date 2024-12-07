from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_plus, vals_minus = list(vals), list(vals)

    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable.

        Args:
        ----
            x (Any): The derivative to be accumulated.

        Returns:
        -------
            None

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf node in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients.

        Args:
        ----
            d_output (Any): The gradient of the output with respect to this variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples containing parent variables and their gradients.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    visited = set()
    order = []

    def dfs(v: Variable) -> None:
        """Performs a depth-first search on the computation graph.

        Args:
        ----
            v (Variable): The current variable being processed.

        Returns:
        -------
            None

        """
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(v.unique_id)
        for parent in v.parents:
            dfs(parent)
        order.append(v)

    dfs(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    """
    # Implement for Task 1.4.
    sorted_variables = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for var in sorted_variables:
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])
        else:
            d = derivatives[var.unique_id]
            for parent, d_input in var.chain_rule(d):
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += d_input
                else:
                    derivatives[parent.unique_id] = d_input


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the values saved for backward computation."""
        return self.saved_values
