"""수치 미분과 그래디언트 시각화 테스트입니다."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.quiver import Quiver

from src.calculus import (
    central_difference,
    numerical_gradient,
    plot_gradient_field,
    quadratic_bowl,
    quadratic_bowl_gradient,
)


def test_central_difference_matches_square_derivative_at_three():
    """중심차분 공식을 바꾸면 도함수 값 6을 놓쳐야 합니다."""
    assert central_difference(lambda value: value**2, 3.0) == pytest.approx(
        6.0,
        abs=1e-4,
    )


def test_numerical_gradient_matches_quadratic_gradient():
    """어느 한 좌표라도 생략하면 알려진 이차 함수 그래디언트와 달라져야 합니다."""
    point = np.array([2.0, -3.0])
    np.testing.assert_allclose(
        numerical_gradient(lambda p: p[0] ** 2 + p[1] ** 2, point),
        [4.0, -6.0],
        atol=1e-4,
    )


def test_gradient_is_perpendicular_to_contour_tangent():
    """해석적 그래디언트가 틀리면 등고선 접선과의 수직성을 잃어야 합니다."""
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
    """잘못된 간격과 빈 점은 그럴듯한 수치를 만들면 안 됩니다."""
    with pytest.raises(ValueError):
        call()


def test_gradient_plot_contains_contours_and_vector_field():
    """등고선이나 화살표를 제거하면 시각화 조건이 깨져야 합니다."""
    figure, axes = plot_gradient_field()
    assert axes.collections
    assert axes.get_aspect() in (1.0, "equal")
    assert axes.get_title() == "Gradient of f(x, y) = x² + y²"
    plt.close(figure)


def test_gradient_plot_marks_tangents_perpendicular_to_every_gradient():
    """어느 점의 접선이나 직각 마커가 빠지면 수직 관계를 읽을 수 없어야 합니다."""
    figure, axes = plot_gradient_field()
    arrows = {
        arrow.get_label(): arrow
        for arrow in axes.collections
        if isinstance(arrow, Quiver)
    }

    gradient_arrow = arrows["Gradient direction"]
    tangent_arrow = arrows["Tangent direction"]
    np.testing.assert_allclose(tangent_arrow.X, gradient_arrow.X)
    np.testing.assert_allclose(tangent_arrow.Y, gradient_arrow.Y)
    dot_products = gradient_arrow.U * tangent_arrow.U + gradient_arrow.V * tangent_arrow.V
    np.testing.assert_allclose(dot_products, 0.0, atol=1e-12)
    assert len(axes.lines) == len(gradient_arrow.X)
    plt.close(figure)


def test_quadratic_bowl_supports_mesh_arrays():
    """벡터화가 깨지면 등고선 격자 계산이 실패해야 합니다."""
    x_values = np.array([[0.0, 1.0]])
    y_values = np.array([[2.0, 3.0]])
    np.testing.assert_allclose(quadratic_bowl(x_values, y_values), [[4.0, 10.0]])
