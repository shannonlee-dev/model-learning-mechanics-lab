"""Tests for linear transformations and matrix decompositions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.linear_algebra import (
    area_scale_error,
    compress_image_svd,
    plot_matrix_transform,
    power_iteration_comparison,
    plot_svd_reconstructions,
    power_iteration,
    rotation_matrix,
    scaling_matrix,
    shear_matrix,
    transform_points,
    unit_circle,
)


def test_standard_transforms_have_expected_geometry():
    """Changing a transform formula must break its known point mapping."""
    point = np.array([[1.0], [0.0]])
    np.testing.assert_allclose(
        rotation_matrix(np.pi / 2) @ point,
        [[0.0], [1.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        scaling_matrix(2.0, 0.5) @ point,
        [[2.0], [0.0]],
    )
    np.testing.assert_allclose(
        shear_matrix(1.5) @ np.array([[0.0], [1.0]]),
        [[1.5], [1.0]],
    )


@pytest.mark.parametrize(
    "matrix",
    [rotation_matrix(0.4), scaling_matrix(2, 0.5), shear_matrix(1.2)],
)
def test_area_ratio_matches_absolute_determinant_within_one_percent(matrix):
    """Breaking point transformation or polygon area must violate det scaling."""
    determinant, ratio, error = area_scale_error(matrix, unit_circle())
    assert error <= 0.01
    assert ratio == pytest.approx(abs(determinant), rel=0.01)


def test_transform_points_rejects_wrong_matrix_shape():
    """Accepting a non-2x2 matrix would make geometric results ambiguous."""
    with pytest.raises(ValueError, match="2x2"):
        transform_points(np.eye(3), unit_circle())


def test_matrix_transform_plot_overlays_original_and_transformed_curves():
    """Dropping either curve must break the visual comparison contract."""
    figure, axes = plot_matrix_transform(scaling_matrix(2.0, 0.5), "Scaling")
    labels = {line.get_label() for line in axes.lines}
    assert {"Original", "Transformed"}.issubset(labels)
    assert axes.get_aspect() in (1.0, "equal")
    plt.close(figure)


def test_power_iteration_matches_dominant_numpy_eigenpair():
    """Wrong normalization or quotient must disagree with the reference pair."""
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    value, vector, iterations = power_iteration(
        matrix,
        initial_vector=np.array([1.0, 1.0]),
    )
    reference_values, reference_vectors = np.linalg.eig(matrix)
    index = np.argmax(np.abs(reference_values))
    assert value == pytest.approx(reference_values[index], rel=0.05)
    assert abs(vector @ reference_vectors[:, index]) == pytest.approx(1.0, abs=1e-5)
    assert 1 <= iterations <= 1000


def test_power_iteration_comparison_reports_numpy_reference_and_small_errors():
    """The submitted comparison must expose eig's reference pair and errors."""
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    comparison = power_iteration_comparison(matrix, initial_vector=np.array([1.0, 1.0]))

    assert comparison["power_eigenvalue"] == pytest.approx(comparison["reference_eigenvalue"])
    assert comparison["eigenvalue_absolute_error"] < 1e-8
    assert comparison["eigenvector_alignment"] == pytest.approx(1.0, abs=1e-8)
    assert comparison["eigenvector_l2_error"] < 1e-8
    assert comparison["iterations"] >= 1


def test_power_iteration_rejects_zero_initial_vector():
    """A zero vector cannot be normalized into an eigenvector estimate."""
    with pytest.raises(ValueError, match="initial vector"):
        power_iteration(np.eye(2), initial_vector=np.zeros(2))


def test_power_iteration_rejects_non_square_matrix():
    """Power Iteration is undefined for non-square matrices."""
    with pytest.raises(ValueError, match="square"):
        power_iteration(np.ones((2, 3)))


def test_svd_rank_reconstruction_error_decreases_with_k():
    """Using more singular triplets must retain more image information."""
    image = np.random.default_rng(42).random((128, 128))
    error_10 = np.linalg.norm(image - compress_image_svd(image, 10))
    error_50 = np.linalg.norm(image - compress_image_svd(image, 50))
    error_100 = np.linalg.norm(image - compress_image_svd(image, 100))
    assert error_100 < error_50 < error_10


@pytest.mark.parametrize("rank", [0, 129])
def test_svd_rejects_rank_outside_image_dimensions(rank):
    """An invalid truncation rank must not be silently clipped."""
    with pytest.raises(ValueError, match="rank"):
        compress_image_svd(np.ones((128, 128)), rank)


def test_svd_comparison_plot_contains_original_and_requested_ranks():
    """Omitting a rank panel must break the requested visual comparison."""
    image = np.diag(np.linspace(1.0, 0.1, 128))
    figure, axes = plot_svd_reconstructions(image)
    assert len(axes) == 4
    assert [axis.get_title() for axis in axes] == ["Original", "k=10", "k=50", "k=100"]
    plt.close(figure)
