"""수치 미분과 2차원 그래디언트 시각화 기능을 제공합니다."""

from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def central_difference(
    function: Callable[[float], float],
    x: float,
    h: float = 1e-5,
) -> float:
    """``(f(x+h)-f(x-h))/(2h)``로 ``f'(x)``를 근사합니다."""
    if h <= 0 or not np.isfinite(h):
        raise ValueError("h must be a positive finite number")
    result = (function(x + h) - function(x - h)) / (2.0 * h)
    result_array = np.asarray(result)
    if result_array.ndim != 0 or not np.isfinite(result_array):
        raise ValueError("function must return one finite scalar")
    return float(result_array)


def numerical_gradient(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """좌표별 중심차분으로 스칼라 함수의 그래디언트를 근사합니다."""
    if h <= 0 or not np.isfinite(h):
        raise ValueError("h must be a positive finite number")
    point_array = np.asarray(point, dtype=float)
    if point_array.ndim != 1 or point_array.size == 0:
        raise ValueError("point must be a non-empty one-dimensional vector")
    if not np.isfinite(point_array).all():
        raise ValueError("point must contain only finite values")

    gradient = np.empty_like(point_array)
    for index in range(point_array.size):
        forward = point_array.copy()
        backward = point_array.copy()
        forward[index] += h
        backward[index] -= h
        forward_value = np.asarray(function(forward))
        backward_value = np.asarray(function(backward))
        if forward_value.ndim != 0 or backward_value.ndim != 0:
            raise ValueError("function must return one scalar")
        gradient[index] = (float(forward_value) - float(backward_value)) / (2.0 * h)
    if not np.isfinite(gradient).all():
        raise ValueError("function must return finite values near the point")
    return gradient


def quadratic_bowl(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """방사형 이차 함수 ``f(x, y) = x² + y²``의 값을 계산합니다."""
    return np.asarray(x) ** 2 + np.asarray(y) ** 2


def quadratic_bowl_gradient(point: np.ndarray) -> np.ndarray:
    """이차 함수의 해석적 그래디언트 ``[2x, 2y]``를 반환합니다."""
    point_array = np.asarray(point, dtype=float)
    if point_array.shape != (2,) or not np.isfinite(point_array).all():
        raise ValueError("point must be a finite vector with shape (2,)")
    return 2.0 * point_array


def plot_gradient_field(
    function: Callable[[np.ndarray, np.ndarray], np.ndarray] = quadratic_bowl,
    gradient: Callable[[np.ndarray], np.ndarray] = quadratic_bowl_gradient,
    sample_points: Optional[np.ndarray] = None,
) -> Tuple[Figure, Axes]:
    """선택한 점에서 2차원 등고선과 정규화한 그래디언트 화살표를 그립니다."""
    coordinates = np.linspace(-4.0, 4.0, 161)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    z_grid = np.asarray(function(x_grid, y_grid), dtype=float)
    if z_grid.shape != x_grid.shape or not np.isfinite(z_grid).all():
        raise ValueError("function must return a finite value for every grid point")

    if sample_points is None:
        samples = np.array(
            [
                [-3.0, -2.0],
                [-2.0, 2.0],
                [1.0, -3.0],
                [2.0, 1.0],
                [3.0, 3.0],
            ]
        )
    else:
        samples = np.asarray(sample_points, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2 or samples.shape[0] == 0:
        raise ValueError("sample_points must have shape (n, 2) with n > 0")
    if not np.isfinite(samples).all():
        raise ValueError("sample_points must contain only finite values")

    vectors = np.vstack([np.asarray(gradient(point), dtype=float) for point in samples])
    if vectors.shape != samples.shape or not np.isfinite(vectors).all():
        raise ValueError("gradient must return one finite vector with shape (2,)")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(lengths == 0.0):
        raise ValueError("gradient arrows require non-zero vectors")
    directions = vectors / lengths

    figure, axes = plt.subplots(figsize=(7, 6))
    contours = axes.contour(x_grid, y_grid, z_grid, levels=16, cmap="viridis")
    axes.clabel(contours, inline=True, fontsize=8)
    axes.quiver(
        samples[:, 0],
        samples[:, 1],
        directions[:, 0],
        directions[:, 1],
        color="tab:red",
        angles="xy",
        scale_units="xy",
        scale=0.45,
        width=0.008,
        label="Gradient direction",
    )
    axes.scatter(samples[:, 0], samples[:, 1], color="black", s=22, zorder=3)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("Gradient of f(x, y) = x² + y²")
    axes.grid(alpha=0.2)
    figure.tight_layout()
    return figure, axes
