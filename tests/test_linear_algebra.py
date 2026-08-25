"""선형 변환과 행렬 분해 테스트입니다."""

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
    """변환 공식을 바꾸면 알려진 점의 대응 관계가 깨져야 합니다."""
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
    """점 변환이나 다각형 넓이 계산이 깨지면 det 스케일링 조건을 위반해야 합니다."""
    determinant, ratio, error = area_scale_error(matrix, unit_circle())
    assert error <= 0.01
    assert ratio == pytest.approx(abs(determinant), rel=0.01)


def test_transform_points_rejects_wrong_matrix_shape():
    """2x2가 아닌 행렬을 허용하면 기하학적 결과가 모호해집니다."""
    with pytest.raises(ValueError, match="2x2"):
        transform_points(np.eye(3), unit_circle())


def test_matrix_transform_plot_overlays_original_and_transformed_curves():
    """두 곡선 중 하나를 빼면 시각적 비교 조건이 깨져야 합니다."""
    figure, axes = plot_matrix_transform(scaling_matrix(2.0, 0.5), "Scaling")
    labels = {line.get_label() for line in axes.lines}
    assert {"Original", "Transformed"}.issubset(labels)
    assert axes.get_aspect() in (1.0, "equal")
    plt.close(figure)


def test_rotation_plot_marks_basis_vectors_and_rotation_angle():
    """회전 대칭인 원에서도 기준 벡터의 이동과 회전각은 보여야 합니다."""
    figure, axes = plot_matrix_transform(
        rotation_matrix(np.pi / 4),
        "Rotation",
    )
    labels = {artist.get_label() for artist in axes.collections}

    assert {"Original basis", "Transformed basis"}.issubset(labels)
    assert any(text.get_text() == "45°" for text in axes.texts)
    assert len(axes.patches) == 1
    plt.close(figure)


def test_power_iteration_matches_dominant_numpy_eigenpair():
    """정규화나 몫이 틀리면 기준 고유쌍과 일치하지 않아야 합니다."""
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
    """제출한 비교 결과는 eig의 기준 고유쌍과 오차를 보여줘야 합니다."""
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    comparison = power_iteration_comparison(matrix, initial_vector=np.array([1.0, 1.0]))

    assert comparison["power_eigenvalue"] == pytest.approx(comparison["reference_eigenvalue"])
    assert comparison["eigenvalue_absolute_error"] < 1e-8
    assert comparison["eigenvector_alignment"] == pytest.approx(1.0, abs=1e-8)
    assert comparison["eigenvector_l2_error"] < 1e-8
    assert comparison["iterations"] >= 1


def test_power_iteration_rejects_zero_initial_vector():
    """영벡터는 고유벡터 추정값으로 정규화할 수 없습니다."""
    with pytest.raises(ValueError, match="initial vector"):
        power_iteration(np.eye(2), initial_vector=np.zeros(2))


def test_power_iteration_rejects_non_square_matrix():
    """Power Iteration은 정사각 행렬이 아니면 정의되지 않습니다."""
    with pytest.raises(ValueError, match="square"):
        power_iteration(np.ones((2, 3)))


def test_svd_rank_reconstruction_error_decreases_with_k():
    """특이 삼중항을 더 많이 쓰면 더 많은 이미지 정보가 남아야 합니다."""
    image = np.random.default_rng(42).random((128, 128))
    error_10 = np.linalg.norm(image - compress_image_svd(image, 10))
    error_50 = np.linalg.norm(image - compress_image_svd(image, 50))
    error_100 = np.linalg.norm(image - compress_image_svd(image, 100))
    assert error_100 < error_50 < error_10


@pytest.mark.parametrize("rank", [0, 129])
def test_svd_rejects_rank_outside_image_dimensions(rank):
    """유효하지 않은 절단 rank는 조용히 잘리면 안 됩니다."""
    with pytest.raises(ValueError, match="rank"):
        compress_image_svd(np.ones((128, 128)), rank)


def test_svd_comparison_plot_contains_original_and_requested_ranks():
    """rank 패널을 누락하면 요구한 시각적 비교가 깨져야 합니다."""
    image = np.diag(np.linspace(1.0, 0.1, 128))
    figure, axes = plot_svd_reconstructions(image)
    assert len(axes) == 4
    assert [axis.get_title() for axis in axes] == ["Original", "k=10", "k=50", "k=100"]
    plt.close(figure)
