"""Probability distributions, stable Softmax, and comparison plots."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def normal_pdf(
    x: np.ndarray,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """Evaluate the normal density ``exp(-z²/2)/(std*sqrt(2π))``."""
    if std <= 0 or not np.isfinite(std):
        raise ValueError("std must be a positive finite number")
    if not np.isfinite(mean):
        raise ValueError("mean must be finite")
    values = np.asarray(x, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("x must contain at least one finite value")
    standardized = (values - mean) / std
    return np.exp(-0.5 * standardized**2) / (std * np.sqrt(2.0 * np.pi))


def bernoulli_pmf(outcomes: np.ndarray, probability: float) -> np.ndarray:
    """Evaluate ``p^x * (1-p)^(1-x)`` for binary outcomes zero or one."""
    if probability < 0 or probability > 1 or not np.isfinite(probability):
        raise ValueError("probability must be finite and lie in [0, 1]")
    outcome_array = np.asarray(outcomes, dtype=float)
    if outcome_array.size == 0 or not np.isfinite(outcome_array).all():
        raise ValueError("outcomes must contain at least one finite value")
    if not np.all(np.isin(outcome_array, [0.0, 1.0])):
        raise ValueError("outcomes must contain only 0 or 1")
    return probability**outcome_array * (1.0 - probability) ** (1.0 - outcome_array)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Normalize one logit vector after subtracting its maximum for stability."""
    logit_array = np.asarray(logits, dtype=float)
    if logit_array.ndim != 1 or logit_array.size == 0:
        raise ValueError("logits must be a non-empty one-dimensional vector")
    if not np.isfinite(logit_array).all():
        raise ValueError("logits must contain only finite values")
    shifted = logit_array - np.max(logit_array)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def plot_normal_distributions() -> Tuple[Figure, Axes]:
    """Plot the PDFs of ``N(0,1)`` and ``N(2,0.5)`` on one figure."""
    x_values = np.linspace(-4.0, 4.5, 500)
    figure, axes = plt.subplots(figsize=(7, 5))
    axes.plot(x_values, normal_pdf(x_values, 0.0, 1.0), label="N(0, 1)")
    axes.plot(x_values, normal_pdf(x_values, 2.0, 0.5), label="N(2, 0.5)")
    axes.set_xlabel("x")
    axes.set_ylabel("Probability density")
    axes.set_title("Normal distribution PDFs")
    axes.legend()
    axes.grid(alpha=0.25)
    figure.tight_layout()
    return figure, axes


def plot_bernoulli_distributions() -> Tuple[Figure, Axes]:
    """Plot the PMFs of ``B(0.3)`` and ``B(0.7)`` on one figure."""
    outcomes = np.array([0, 1])
    width = 0.34
    figure, axes = plt.subplots(figsize=(6, 5))
    axes.bar(
        outcomes - width / 2,
        bernoulli_pmf(outcomes, 0.3),
        width=width,
        label="B(0.3)",
    )
    axes.bar(
        outcomes + width / 2,
        bernoulli_pmf(outcomes, 0.7),
        width=width,
        label="B(0.7)",
    )
    axes.set_xticks(outcomes)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("Outcome")
    axes.set_ylabel("Probability mass")
    axes.set_title("Bernoulli distribution PMFs")
    axes.legend()
    axes.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure, axes
