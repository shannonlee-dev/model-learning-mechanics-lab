"""Linear transformations, eigenvalue estimation, and SVD compression."""

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def unit_circle(num_points: int = 361) -> np.ndarray:
    """Return evenly sampled unit-circle points with shape ``(2, n)``."""
    if not isinstance(num_points, int) or num_points < 3:
        raise ValueError("num_points must be an integer of at least 3")
    angles = np.linspace(0.0, 2.0 * np.pi, num_points)
    return np.vstack((np.cos(angles), np.sin(angles)))


def rotation_matrix(theta: float) -> np.ndarray:
    """Return the 2D counter-clockwise rotation matrix ``R(theta)``."""
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=float)


def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    """Return the 2D scaling matrix ``diag(sx, sy)``."""
    return np.array([[sx, 0.0], [0.0, sy]], dtype=float)


def shear_matrix(k: float) -> np.ndarray:
    """Return the horizontal shear matrix ``[[1, k], [0, 1]]``."""
    return np.array([[1.0, k], [0.0, 1.0]], dtype=float)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 2×2 linear transformation to ``(2, n)`` point columns."""
    matrix_array = np.asarray(matrix, dtype=float)
    point_array = np.asarray(points, dtype=float)
    if matrix_array.shape != (2, 2):
        raise ValueError("matrix must have shape 2x2")
    if point_array.ndim != 2 or point_array.shape[0] != 2 or point_array.shape[1] == 0:
        raise ValueError("points must have shape (2, n) with n > 0")
    if not np.isfinite(matrix_array).all() or not np.isfinite(point_array).all():
        raise ValueError("matrix and points must contain only finite values")
    return matrix_array @ point_array


def polygon_area(points: np.ndarray) -> float:
    """Compute polygon area with the shoelace formula for ``(2, n)`` points."""
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[0] != 2 or point_array.shape[1] < 3:
        raise ValueError("points must have shape (2, n) with n >= 3")
    if not np.isfinite(point_array).all():
        raise ValueError("points must contain only finite values")
    x_values, y_values = point_array
    twice_area = np.dot(x_values, np.roll(y_values, -1)) - np.dot(
        y_values,
        np.roll(x_values, -1),
    )
    return float(abs(twice_area) / 2.0)


def area_scale_error(
    matrix: np.ndarray,
    points: np.ndarray,
) -> Tuple[float, float, float]:
    """Compare ``abs(det(A))`` with the transformed-to-original area ratio."""
    matrix_array = np.asarray(matrix, dtype=float)
    transformed = transform_points(matrix_array, points)
    original_area = polygon_area(points)
    if original_area == 0.0:
        raise ValueError("points must enclose a non-zero area")
    transformed_area = polygon_area(transformed)
    determinant = float(np.linalg.det(matrix_array))
    area_ratio = transformed_area / original_area
    expected_ratio = abs(determinant)
    if expected_ratio == 0.0:
        relative_error = abs(area_ratio - expected_ratio)
    else:
        relative_error = abs(area_ratio - expected_ratio) / expected_ratio
    return determinant, float(area_ratio), float(relative_error)


def plot_matrix_transform(
    matrix: np.ndarray,
    title: str,
    points: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Overlay original and transformed unit-circle points on one figure."""
    point_array = unit_circle() if points is None else np.asarray(points, dtype=float)
    transformed = transform_points(matrix, point_array)
    if ax is None:
        figure, axes = plt.subplots(figsize=(6, 6))
    else:
        axes = ax
        figure = axes.figure
    axes.plot(point_array[0], point_array[1], label="Original", color="tab:blue")
    axes.plot(
        transformed[0],
        transformed[1],
        label="Transformed",
        color="tab:red",
    )
    axes.axhline(0.0, color="black", linewidth=0.5)
    axes.axvline(0.0, color="black", linewidth=0.5)
    axes.set_aspect("equal", adjustable="box")
    axes.set_title(title)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.tight_layout()
    return figure, axes


def power_iteration(
    matrix: np.ndarray,
    initial_vector: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-9,
) -> Tuple[float, np.ndarray, int]:
    """Estimate the dominant eigenpair using normalized Power Iteration.

    The eigenvalue estimate is the Rayleigh quotient ``v.T @ A @ v``.
    """
    matrix_array = np.asarray(matrix, dtype=float)
    if (
        matrix_array.ndim != 2
        or matrix_array.shape[0] == 0
        or matrix_array.shape[0] != matrix_array.shape[1]
    ):
        raise ValueError("matrix must be a non-empty square matrix")
    if not np.isfinite(matrix_array).all():
        raise ValueError("matrix must contain only finite values")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a positive finite number")

    dimension = matrix_array.shape[0]
    vector = (
        np.ones(dimension, dtype=float)
        if initial_vector is None
        else np.asarray(initial_vector, dtype=float).copy()
    )
    if vector.shape != (dimension,):
        raise ValueError("initial vector must match the matrix dimension")
    if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0.0:
        raise ValueError("initial vector must be finite and non-zero")
    vector /= np.linalg.norm(vector)

    for iteration in range(1, max_iterations + 1):
        multiplied = matrix_array @ vector
        norm = np.linalg.norm(multiplied)
        if norm == 0.0:
            raise ValueError("matrix maps the current vector to zero")
        next_vector = multiplied / norm
        direction_change = min(
            np.linalg.norm(next_vector - vector),
            np.linalg.norm(next_vector + vector),
        )
        vector = next_vector
        if direction_change <= tolerance:
            eigenvalue = float(vector @ matrix_array @ vector)
            return eigenvalue, vector, iteration
    raise RuntimeError("power iteration did not converge within max_iterations")


def power_iteration_comparison(
    matrix: np.ndarray,
    initial_vector: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-9,
) -> Dict[str, object]:
    """Compare Power Iteration's dominant eigenpair against ``np.linalg.eig``.

    Eigenvectors are sign-invariant, so the reported L2 error aligns the
    reference vector's sign with the Power Iteration vector first.
    """
    power_value, power_vector, iterations = power_iteration(
        matrix,
        initial_vector=initial_vector,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    reference_values, reference_vectors = np.linalg.eig(np.asarray(matrix, dtype=float))
    index = int(np.argmax(np.abs(reference_values)))
    reference_value = float(np.real_if_close(reference_values[index]))
    reference_vector = np.asarray(np.real_if_close(reference_vectors[:, index]), dtype=float)
    reference_vector /= np.linalg.norm(reference_vector)
    if power_vector @ reference_vector < 0.0:
        reference_vector = -reference_vector

    return {
        "power_eigenvalue": power_value,
        "power_eigenvector": power_vector,
        "reference_eigenvalue": reference_value,
        "reference_eigenvector": reference_vector,
        "eigenvalue_absolute_error": abs(power_value - reference_value),
        "eigenvector_alignment": abs(float(power_vector @ reference_vector)),
        "eigenvector_l2_error": float(np.linalg.norm(power_vector - reference_vector)),
        "iterations": iterations,
    }


def compress_image_svd(image: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct a grayscale image from its first ``k`` singular triplets."""
    image_array = np.asarray(image, dtype=float)
    if image_array.ndim != 2 or image_array.size == 0:
        raise ValueError("image must be a non-empty 2D grayscale array")
    if not np.isfinite(image_array).all():
        raise ValueError("image must contain only finite values")
    maximum_rank = min(image_array.shape)
    if not isinstance(k, (int, np.integer)) or not 1 <= int(k) <= maximum_rank:
        raise ValueError(f"rank must be between 1 and {maximum_rank}")
    rank = int(k)
    left, singular_values, right_transpose = np.linalg.svd(
        image_array,
        full_matrices=False,
    )
    return (left[:, :rank] * singular_values[:rank]) @ right_transpose[:rank, :]


def plot_svd_reconstructions(
    image: np.ndarray,
    ranks: Sequence[int] = (10, 50, 100),
) -> Tuple[Figure, np.ndarray]:
    """Compare the original grayscale image with truncated-SVD reconstructions."""
    rank_values = tuple(ranks)
    if not rank_values:
        raise ValueError("ranks must contain at least one rank")
    reconstructions = [compress_image_svd(image, rank) for rank in rank_values]
    figure, axes = plt.subplots(1, len(rank_values) + 1, figsize=(4 * (len(rank_values) + 1), 4))
    axes_array = np.atleast_1d(axes)
    panels = [np.asarray(image, dtype=float)] + reconstructions
    titles = ["Original"] + [f"k={rank}" for rank in rank_values]
    for axis, panel, panel_title in zip(axes_array, panels, titles):
        axis.imshow(panel, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(panel_title)
        axis.axis("off")
    figure.tight_layout()
    return figure, axes_array
