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
    plot_path_metric,
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


def test_path_plot_identifies_start_optimum_and_step_direction():
    """기준점이나 진행 방향이 빠지면 경로를 시간 순서로 읽을 수 없어야 합니다."""
    path = np.array([[2.0, 2.0], [1.0, 1.0], [0.5, 0.5]])
    figure, axes = plot_optimization_paths(
        sphere,
        {"Vanilla GD": path},
        x_limits=(-3.0, 3.0),
        y_limits=(-3.0, 3.0),
        title="Optimizer path",
    )

    legend_labels = {text.get_text() for text in axes.get_legend().get_texts()}
    direction_arrows = [
        annotation
        for annotation in axes.texts
        if getattr(annotation, "arrow_patch", None) is not None
    ]
    assert {"Start", "Optimum"}.issubset(legend_labels)
    assert direction_arrows
    plt.close(figure)


def test_path_color_stays_consistent_when_a_series_moves_to_its_own_panel():
    """동일한 발산 경로가 독립 패널에서 다른 색으로 바뀌면 비교가 깨져야 합니다."""
    convergent = np.array([[2.0, 2.0], [1.0, 1.0]])
    divergent = np.array([[2.0, 2.0], [-3.0, -3.0]])
    combined_figure, combined_axes = plot_optimization_paths(
        sphere,
        {"lr=0.1 (convergent)": convergent, "lr=1.1 (divergent)": divergent},
        x_limits=(-4.0, 4.0),
        y_limits=(-4.0, 4.0),
        title="Combined",
    )
    separate_figure, separate_axes = plot_optimization_paths(
        sphere,
        {"lr=1.1 (divergent)": divergent},
        x_limits=(-4.0, 4.0),
        y_limits=(-4.0, 4.0),
        title="Separate",
    )

    combined_color = next(
        line.get_color()
        for line in combined_axes.lines
        if line.get_label() == "lr=1.1 (divergent)"
    )
    separate_color = separate_axes.lines[0].get_color()
    assert separate_color == combined_color
    plt.close(combined_figure)
    plt.close(separate_figure)


def test_path_metric_plot_compares_named_paths_on_log_scale():
    """손실 진단 패널이나 로그 축을 제거하면 optimizer 비교가 약해져야 합니다."""
    paths = {
        "Vanilla GD": np.array([[2.0, 2.0], [1.0, 1.0], [0.5, 0.5]]),
        "Momentum": np.array([[2.0, 2.0], [0.7, 1.1], [0.2, 0.3]]),
    }

    figure, axes = plot_path_metric(
        paths,
        metric=sphere,
        title="Objective value by iteration",
        ylabel="Objective f(xₖ)",
    )

    assert axes.get_yscale() == "log"
    assert axes.get_xlabel() == "Iteration"
    assert axes.get_ylabel() == "Objective f(xₖ)"
    assert {line.get_label() for line in axes.lines} == {"Vanilla GD", "Momentum"}
    plt.close(figure)
