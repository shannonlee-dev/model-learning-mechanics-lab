"""Tests for probability distributions and stable Softmax."""

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
    """Changing normalization must miss the known density at zero."""
    density = normal_pdf(np.array([0.0]))[0]
    assert density == pytest.approx(1.0 / np.sqrt(2.0 * np.pi))


def test_normal_pdf_reflects_mean_and_standard_deviation():
    """Ignoring mean or standard deviation must break these literal densities."""
    values = normal_pdf(np.array([2.0, 2.5]), mean=2.0, std=0.5)
    expected = np.array([0.7978845608, 0.4839414490])
    np.testing.assert_allclose(values, expected, rtol=1e-9)


def test_bernoulli_pmf_returns_failure_and_success_probabilities():
    """Swapping outcomes must invert the requested Bernoulli masses."""
    np.testing.assert_allclose(
        bernoulli_pmf(np.array([0, 1]), 0.3),
        [0.7, 0.3],
    )


def test_softmax_is_stable_and_normalized():
    """Removing max subtraction must overflow for these large logits."""
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
    """Invalid distributions and empty logits must fail visibly."""
    with pytest.raises(ValueError):
        call()


def test_distribution_plots_include_both_requested_parameter_sets():
    """Dropping a requested distribution must remove its legend label."""
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
