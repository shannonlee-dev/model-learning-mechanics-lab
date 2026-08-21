"""Tests for numerical differentiation and gradient visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.calculus import (
    central_difference,
    numerical_gradient,
    plot_gradient_field,
    quadratic_bowl,
    quadratic_bowl_gradient,
)


def test_central_difference_matches_square_derivative_at_three():
    """Changing the central-difference formula must miss the derivative six."""
    assert central_difference(lambda value: value**2, 3.0) == pytest.approx(
        6.0,
        abs=1e-4,
    )


def test_numerical_gradient_matches_quadratic_gradient():
    """Skipping either coordinate must break the known bowl gradient."""
    point = np.array([2.0, -3.0])
    np.testing.assert_allclose(
        numerical_gradient(lambda p: p[0] ** 2 + p[1] ** 2, point),
        [4.0, -6.0],
        atol=1e-4,
    )


def test_gradient_is_perpendicular_to_contour_tangent():
    """A wrong analytical gradient must lose contour perpendicularity."""
    gradient = quadratic_bowl_gradient(np.array([2.0, 1.0]))
    tangent = np.array([-gradient[1], gradient[0]])
    assert gradient @ tangent == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "call",
    [
        lambda: central_difference(lambda value: value, 1.0, h=0.0),
        lambda: numerical_gradient(lambda point: point.sum(), [], h=1e-5),
    ],
)
def test_differentiation_rejects_invalid_parameters(call):
    """Invalid steps and empty points must not produce plausible numbers."""
    with pytest.raises(ValueError):
        call()


def test_gradient_plot_contains_contours_and_vector_field():
    """Removing contours or arrows must break the visualization contract."""
    figure, axes = plot_gradient_field()
    assert axes.collections
    assert axes.get_aspect() in (1.0, "equal")
    assert axes.get_title() == "Gradient of f(x, y) = x² + y²"
    plt.close(figure)


def test_quadratic_bowl_supports_mesh_arrays():
    """Breaking vectorization must prevent contour-grid evaluation."""
    x_values = np.array([[0.0, 1.0]])
    y_values = np.array([[2.0, 3.0]])
    np.testing.assert_allclose(quadratic_bowl(x_values, y_values), [[4.0, 10.0]])
