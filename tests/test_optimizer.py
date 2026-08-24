"""경사하강법 옵티마이저와 수렴 경로 테스트입니다."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.optimizer import (
    Momentum,
    VanillaGD,
    elliptical,
    elliptical_gradient,
    optimize,
    plot_optimization_paths,
    sphere,
    sphere_gradient,
)


def test_vanilla_gd_applies_gradient_step():
    """갱신 부호나 학습률 계수를 바꾸면 이 단계가 깨져야 합니다."""
    updated = VanillaGD(0.1).update(
        np.array([2.0, -1.0]),
        np.array([4.0, -2.0]),
    )
    np.testing.assert_allclose(updated, [1.6, -0.8])


def test_momentum_accumulates_velocity():
    """저장한 속도를 없애면 두 번째 갱신값이 틀려야 합니다."""
    optimizer = Momentum(learning_rate=0.1, beta=0.9)
    first = optimizer.update(np.array([1.0]), np.array([2.0]))
    second = optimizer.update(first, np.array([2.0]))
    np.testing.assert_allclose(first, [0.8])
    np.testing.assert_allclose(second, [0.42])


@pytest.mark.parametrize(
    "factory",
    [lambda: VanillaGD(0), lambda: Momentum(0.1, beta=1.0)],
)
def test_optimizers_reject_invalid_hyperparameters(factory):
    """잘못된 하이퍼파라미터는 불안정한 상태를 만들지 말고 실패해야 합니다."""
    with pytest.raises(ValueError):
        factory()


def test_vanilla_gd_reaches_radius_point_one_after_100_steps():
    """경로 반복문이 깨지면 미션의 수렴 임계값을 만족하지 못해야 합니다."""
    path = optimize(sphere_gradient, [5.0, 5.0], VanillaGD(0.1), steps=100)
    assert path.shape == (101, 2)
    assert np.linalg.norm(path[-1]) <= 0.1


def test_learning_rate_one_point_one_diverges_on_sphere():
    """갱신값을 자르거나 바꾸면 요구한 발산이 사라져야 합니다."""
    path = optimize(sphere_gradient, [5.0, 5.0], VanillaGD(1.1), steps=20)
    assert np.linalg.norm(path[-1]) > np.linalg.norm(path[0])


def test_momentum_advances_faster_on_elliptical_bowl():
    """Momentum 누적을 없애면 완만한 축에서의 이점이 사라져야 합니다."""
    vanilla_path = optimize(
        elliptical_gradient,
        [5.0, 5.0],
        VanillaGD(0.01),
        steps=60,
    )
    momentum_path = optimize(
        elliptical_gradient,
        [5.0, 5.0],
        Momentum(0.01, 0.9),
        steps=60,
    )
    assert elliptical(momentum_path[-1]) < elliptical(vanilla_path[-1])


def test_objectives_and_gradients_match_literal_values():
    """이차 함수 계수 중 하나를 바꾸면 알려진 값이 달라져야 합니다."""
    point = np.array([2.0, -3.0])
    assert sphere(point) == pytest.approx(13.0)
    np.testing.assert_allclose(sphere_gradient(point), [4.0, -6.0])
    assert elliptical(point) == pytest.approx(94.0)
    np.testing.assert_allclose(elliptical_gradient(point), [4.0, -60.0])


def test_path_plot_overlays_named_optimizer_paths():
    """이름이 지정된 경로를 빼면 비교 Figure가 깨져야 합니다."""
    paths = {
        "Vanilla GD": np.array([[2.0, 2.0], [1.0, 1.0], [0.5, 0.5]]),
        "Momentum": np.array([[2.0, 2.0], [0.7, 1.1], [0.2, 0.3]]),
    }
    figure, axes = plot_optimization_paths(
        sphere,
        paths,
        x_limits=(-3.0, 3.0),
        y_limits=(-3.0, 3.0),
        title="Optimizer paths",
    )
    labels = {line.get_label() for line in axes.lines}
    assert {"Vanilla GD", "Momentum"}.issubset(labels)
    plt.close(figure)
