"""Gradient-descent optimizers, quadratic objectives, and path plots."""

from typing import Callable, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


class VanillaGD:
    """Vanilla gradient descent with ``theta <- theta - lr * gradient``."""

    def __init__(self, learning_rate: float) -> None:
        """Create a gradient-descent optimizer with a positive learning rate."""
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be a positive finite number")
        self.learning_rate = float(learning_rate)

    def update(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Return parameters after one vanilla gradient-descent step."""
        parameter_array = _as_vector(parameters, "parameters")
        gradient_array = _as_vector(gradient, "gradient")
        if gradient_array.shape != parameter_array.shape:
            raise ValueError("gradient must have the same shape as parameters")
        return parameter_array - self.learning_rate * gradient_array


class Momentum:
    """Gradient descent with accumulated velocity ``v = beta*v + gradient``."""

    def __init__(self, learning_rate: float, beta: float = 0.9) -> None:
        """Create Momentum with positive learning rate and ``0 <= beta < 1``."""
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be a positive finite number")
        if beta < 0 or beta >= 1 or not np.isfinite(beta):
            raise ValueError("beta must be finite and satisfy 0 <= beta < 1")
        self.learning_rate = float(learning_rate)
        self.beta = float(beta)
        self.velocity: Optional[np.ndarray] = None

    def update(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Return parameters after one Momentum step and retain its velocity."""
        parameter_array = _as_vector(parameters, "parameters")
        gradient_array = _as_vector(gradient, "gradient")
        if gradient_array.shape != parameter_array.shape:
            raise ValueError("gradient must have the same shape as parameters")
        if self.velocity is None:
            self.velocity = np.zeros_like(parameter_array)
        elif self.velocity.shape != parameter_array.shape:
            raise ValueError("parameter shape must stay constant across Momentum updates")
        self.velocity = self.beta * self.velocity + gradient_array
        return parameter_array - self.learning_rate * self.velocity


def optimize(
    gradient_function: Callable[[np.ndarray], np.ndarray],
    initial_point: np.ndarray,
    optimizer: Union[VanillaGD, Momentum],
    steps: int,
) -> np.ndarray:
    """Run an optimizer and return a path containing the initial point."""
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    current = _as_vector(initial_point, "initial_point").copy()
    path = [current.copy()]
    for _ in range(steps):
        gradient = np.asarray(gradient_function(current), dtype=float)
        current = optimizer.update(current, gradient)
        path.append(current.copy())
    return np.vstack(path)


def sphere(point: np.ndarray) -> Union[float, np.ndarray]:
    """Evaluate ``f(x, y) = x² + y²`` at points ending in dimension two."""
    point_array = np.asarray(point, dtype=float)
    if point_array.shape[-1:] != (2,) or not np.isfinite(point_array).all():
        raise ValueError("point must be finite and end with dimension 2")
    result = np.sum(point_array**2, axis=-1)
    return float(result) if result.ndim == 0 else result


def sphere_gradient(point: np.ndarray) -> np.ndarray:
    """Return ``[2x, 2y]``, the gradient of the radial quadratic."""
    point_array = _as_vector(point, "point")
    if point_array.shape != (2,):
        raise ValueError("point must have shape (2,)")
    return 2.0 * point_array


def elliptical(point: np.ndarray) -> Union[float, np.ndarray]:
    """Evaluate ``f(x, y) = x² + 10y²`` at dimension-two points."""
    point_array = np.asarray(point, dtype=float)
    if point_array.shape[-1:] != (2,) or not np.isfinite(point_array).all():
        raise ValueError("point must be finite and end with dimension 2")
    result = point_array[..., 0] ** 2 + 10.0 * point_array[..., 1] ** 2
    return float(result) if result.ndim == 0 else result


def elliptical_gradient(point: np.ndarray) -> np.ndarray:
    """Return ``[2x, 20y]``, the gradient of the elliptical quadratic."""
    point_array = _as_vector(point, "point")
    if point_array.shape != (2,):
        raise ValueError("point must have shape (2,)")
    return np.array([2.0 * point_array[0], 20.0 * point_array[1]])


def plot_optimization_paths(
    objective: Callable[[np.ndarray], Union[float, np.ndarray]],
    paths: Mapping[str, np.ndarray],
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    title: str,
    levels: int = 24,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Overlay named optimization paths on objective-function contours."""
    if not paths:
        raise ValueError("paths must contain at least one named path")
    if levels <= 0:
        raise ValueError("levels must be positive")
    x_values = np.linspace(x_limits[0], x_limits[1], 241)
    y_values = np.linspace(y_limits[0], y_limits[1], 241)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid_points = np.stack((x_grid, y_grid), axis=-1)
    z_grid = np.asarray(objective(grid_points), dtype=float)
    if z_grid.shape != x_grid.shape or not np.isfinite(z_grid).all():
        raise ValueError("objective must return one finite value per 2D grid point")

    if ax is None:
        figure, axes = plt.subplots(figsize=(7, 6))
    else:
        axes = ax
        figure = axes.figure
    axes.contour(x_grid, y_grid, z_grid, levels=levels, cmap="viridis", alpha=0.75)

    for label, path in paths.items():
        path_array = np.asarray(path, dtype=float)
        if (
            path_array.ndim != 2
            or path_array.shape[0] == 0
            or path_array.shape[1] != 2
            or not np.isfinite(path_array).all()
        ):
            raise ValueError("each path must be a finite array with shape (n, 2)")
        axes.plot(
            path_array[:, 0],
            path_array[:, 1],
            linestyle="--",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=label,
        )
        axes.scatter(*path_array[0], marker="s", s=45, zorder=4)
        axes.scatter(*path_array[-1], marker="*", s=90, zorder=4)

    axes.set_xlim(x_limits)
    axes.set_ylim(y_limits)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title(title)
    axes.legend()
    axes.grid(alpha=0.2)
    figure.tight_layout()
    return figure, axes
