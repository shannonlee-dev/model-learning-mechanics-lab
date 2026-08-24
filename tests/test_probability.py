"""확률분포와 수치적으로 안정적인 Softmax 테스트입니다."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.probability import (
    bernoulli_pmf,
    normal_pdf,
    plot_bernoulli_distributions,
    plot_normal_distributions,
    softmax,
)


def test_standard_normal_pdf_at_zero():
    """정규화를 바꾸면 0에서의 알려진 밀도값을 놓쳐야 합니다."""
    density = normal_pdf(np.array([0.0]))[0]
    assert density == pytest.approx(1.0 / np.sqrt(2.0 * np.pi))


def test_normal_pdf_reflects_mean_and_standard_deviation():
    """평균이나 표준편차를 무시하면 이 명시적 밀도값과 달라져야 합니다."""
    values = normal_pdf(np.array([2.0, 2.5]), mean=2.0, std=0.5)
    expected = np.array([0.7978845608, 0.4839414490])
    np.testing.assert_allclose(values, expected, rtol=1e-9)


def test_bernoulli_pmf_returns_failure_and_success_probabilities():
    """결과를 뒤바꾸면 요구한 베르누이 확률질량이 반전돼야 합니다."""
    np.testing.assert_allclose(
        bernoulli_pmf(np.array([0, 1]), 0.3),
        [0.7, 0.3],
    )


def test_softmax_is_stable_and_normalized():
    """최댓값 빼기를 제거하면 큰 logit에서 오버플로가 발생해야 합니다."""
    probabilities = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.isfinite(probabilities).all()
    assert abs(probabilities.sum() - 1.0) <= 1e-6
    assert np.argmax(probabilities) == 2


@pytest.mark.parametrize(
    "call",
    [
        lambda: normal_pdf([0.0], std=0.0),
        lambda: bernoulli_pmf([0], -0.1),
        lambda: bernoulli_pmf([2], 0.5),
        lambda: softmax([]),
    ],
)
def test_probability_functions_reject_invalid_parameters(call):
    """잘못된 분포와 빈 logit은 명확히 실패해야 합니다."""
    with pytest.raises(ValueError):
        call()


def test_distribution_plots_include_both_requested_parameter_sets():
    """요구한 분포를 빼면 범례 레이블이 사라져야 합니다."""
    normal_figure, normal_axes = plot_normal_distributions()
    bernoulli_figure, bernoulli_axes = plot_bernoulli_distributions()
    assert {line.get_label() for line in normal_axes.lines} == {
        "N(0, 1)",
        "N(2, 0.5)",
    }
    legend_labels = {
        text.get_text() for text in bernoulli_axes.get_legend().get_texts()
    }
    assert legend_labels == {"B(0.3)", "B(0.7)"}
    plt.close(normal_figure)
    plt.close(bernoulli_figure)
