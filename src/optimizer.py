"""경사하강법 옵티마이저, 이차 목적 함수, 경로 시각화 기능을 제공합니다."""

from typing import Callable, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


PATH_COLORS = ("#2563EB", "#E4572E", "#7C3AED", "#0891B2")


def _path_color(label: str, fallback_index: int) -> str:
    """패널 위치와 무관하게 optimizer 또는 안정성 의미에 맞는 색을 반환합니다."""
    normalized = label.casefold()
    if "momentum" in normalized or "divergent" in normalized:
        return PATH_COLORS[1]
    if "vanilla" in normalized or "convergent" in normalized:
        return PATH_COLORS[0]
    return PATH_COLORS[fallback_index % len(PATH_COLORS)]


def _direction_segments(path: np.ndarray, count: int = 4) -> np.ndarray:
    """경로의 누적 이동 거리에 고르게 분포한 화살표 segment 인덱스를 반환합니다."""
    if path.shape[0] < 2:
        return np.array([], dtype=int)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= np.finfo(float).eps:
        return np.array([], dtype=int)
    cumulative = np.cumsum(lengths)
    targets = np.linspace(0.15, 0.85, min(count, path.shape[0] - 1)) * total
    indices = np.searchsorted(cumulative, targets)
    indices = np.clip(indices, 0, path.shape[0] - 2)
    return np.unique(indices[lengths[indices] > np.finfo(float).eps])


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    """``value``를 유한한 비어 있지 않은 1차원 실수 배열로 변환합니다."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


class VanillaGD:
    """``theta <- theta - lr * gradient``를 사용하는 기본 경사하강법입니다."""

    def __init__(self, learning_rate: float) -> None:
        """양의 학습률을 사용하는 경사하강법 옵티마이저를 생성합니다."""
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be a positive finite number")
        self.learning_rate = float(learning_rate)

    def update(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """기본 경사하강법을 한 번 적용한 파라미터를 반환합니다."""
        parameter_array = _as_vector(parameters, "parameters")
        gradient_array = _as_vector(gradient, "gradient")
        if gradient_array.shape != parameter_array.shape:
            raise ValueError("gradient must have the same shape as parameters")
        return parameter_array - self.learning_rate * gradient_array


class Momentum:
    """``v = beta*v + gradient``의 누적 속도를 사용하는 경사하강법입니다."""

    def __init__(self, learning_rate: float, beta: float = 0.9) -> None:
        """양의 학습률과 ``0 <= beta < 1``을 사용하는 Momentum을 생성합니다."""
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be a positive finite number")
        if beta < 0 or beta >= 1 or not np.isfinite(beta):
            raise ValueError("beta must be finite and satisfy 0 <= beta < 1")
        self.learning_rate = float(learning_rate)
        self.beta = float(beta)
        self.velocity: Optional[np.ndarray] = None

    def update(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Momentum을 한 번 적용하고 속도를 유지한 파라미터를 반환합니다."""
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
    """옵티마이저를 실행하고 초기점을 포함한 경로를 반환합니다."""
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
    """마지막 차원이 2인 점에서 ``f(x, y) = x² + y²``를 계산합니다."""
    point_array = np.asarray(point, dtype=float)
    if point_array.shape[-1:] != (2,) or not np.isfinite(point_array).all():
        raise ValueError("point must be finite and end with dimension 2")
    result = np.sum(point_array**2, axis=-1)
    return float(result) if result.ndim == 0 else result


def sphere_gradient(point: np.ndarray) -> np.ndarray:
    """방사형 이차 함수의 그래디언트 ``[2x, 2y]``를 반환합니다."""
    point_array = _as_vector(point, "point")
    if point_array.shape != (2,):
        raise ValueError("point must have shape (2,)")
    return 2.0 * point_array


def elliptical(point: np.ndarray) -> Union[float, np.ndarray]:
    """2차원 점에서 ``f(x, y) = x² + 10y²``를 계산합니다."""
    point_array = np.asarray(point, dtype=float)
    if point_array.shape[-1:] != (2,) or not np.isfinite(point_array).all():
        raise ValueError("point must be finite and end with dimension 2")
    result = point_array[..., 0] ** 2 + 10.0 * point_array[..., 1] ** 2
    return float(result) if result.ndim == 0 else result


def elliptical_gradient(point: np.ndarray) -> np.ndarray:
    """타원형 이차 함수의 그래디언트 ``[2x, 20y]``를 반환합니다."""
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
    optimum: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[Figure, Axes]:
    """목적 함수 등고선 위에 이름이 지정된 최적화 경로를 겹쳐 그립니다."""
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
    positive_values = z_grid[z_grid > 0]
    if positive_values.size:
        level_min = max(float(positive_values.min()), float(z_grid.max()) * 1e-4)
        contour_levels = np.geomspace(level_min, float(z_grid.max()), levels)
    else:
        contour_levels = levels
    axes.contourf(
        x_grid,
        y_grid,
        z_grid,
        levels=contour_levels,
        cmap="Purples",
        alpha=0.12,
    )
    axes.contour(
        x_grid,
        y_grid,
        z_grid,
        levels=contour_levels,
        colors="#6D477F",
        linewidths=0.9,
        alpha=0.62,
    )

    starts = []
    for index, (label, path) in enumerate(paths.items()):
        path_array = np.asarray(path, dtype=float)
        if (
            path_array.ndim != 2
            or path_array.shape[0] == 0
            or path_array.shape[1] != 2
            or not np.isfinite(path_array).all()
        ):
            raise ValueError("each path must be a finite array with shape (n, 2)")
        color = _path_color(label, index)
        marker_indices = np.unique(
            np.linspace(0, path_array.shape[0] - 1, min(18, path_array.shape[0]), dtype=int)
        )
        axes.plot(
            path_array[:, 0],
            path_array[:, 1],
            color=color,
            linestyle=(0, (4, 3)),
            marker="o",
            markevery=marker_indices,
            markersize=3.5,
            linewidth=1.8,
            label=label,
            zorder=4,
        )
        for segment in _direction_segments(path_array):
            axes.annotate(
                "",
                xy=path_array[segment + 1],
                xytext=path_array[segment],
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 1.5,
                    "mutation_scale": 11,
                },
                zorder=5,
            )
        starts.append(path_array[0])
        axes.scatter(
            *path_array[-1],
            marker="X",
            s=58,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=6,
        )

    for index, start in enumerate(starts):
        axes.scatter(
            *start,
            marker="s",
            s=62,
            color="#16A34A",
            edgecolor="white",
            linewidth=0.8,
            label="Start" if index == 0 else None,
            zorder=7,
        )
    axes.scatter(
        *optimum,
        marker="*",
        s=145,
        color="#D97706",
        edgecolor="white",
        linewidth=0.7,
        label="Optimum",
        zorder=8,
    )

    axes.set_xlim(x_limits)
    axes.set_ylim(y_limits)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title(title)
    axes.legend()
    axes.set_facecolor("#FCFCFD")
    axes.grid(color="#CBD5E1", alpha=0.35, linewidth=0.7)
    if ax is None:
        figure.tight_layout()
    return figure, axes


def plot_path_metric(
    paths: Mapping[str, np.ndarray],
    metric: Callable[[np.ndarray], Union[float, np.ndarray]],
    title: str,
    ylabel: str,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """각 최적화 경로의 지표를 iteration에 따라 로그 축으로 비교합니다."""
    if not paths:
        raise ValueError("paths must contain at least one named path")
    if ax is None:
        figure, axes = plt.subplots(figsize=(6, 4))
    else:
        axes = ax
        figure = axes.figure

    for index, (label, path) in enumerate(paths.items()):
        path_array = np.asarray(path, dtype=float)
        values = np.asarray(metric(path_array), dtype=float)
        if values.shape != (path_array.shape[0],) or not np.isfinite(values).all():
            raise ValueError("metric must return one finite value per path point")
        marker_indices = np.unique(
            np.linspace(0, path_array.shape[0] - 1, min(14, path_array.shape[0]), dtype=int)
        )
        axes.plot(
            np.arange(path_array.shape[0]),
            np.maximum(values, np.finfo(float).tiny),
            color=_path_color(label, index),
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            markevery=marker_indices,
            label=label,
        )

    axes.set_yscale("log")
    axes.set_xlabel("Iteration")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()
    axes.set_facecolor("#FCFCFD")
    axes.grid(color="#CBD5E1", alpha=0.45, linewidth=0.7, which="both")
    if ax is None:
        figure.tight_layout()
    return figure, axes
