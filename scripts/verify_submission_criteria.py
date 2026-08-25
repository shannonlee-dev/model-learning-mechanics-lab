"""제출 기준의 수치 검증 결과를 콘솔에 출력합니다."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "model-learning-mechanics-matplotlib"),
)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.calculus import central_difference
from src.linear_algebra import area_scale_error, power_iteration_comparison, unit_circle
from src.probability import softmax


AREA_ERROR_THRESHOLD = 0.01
EIGENVALUE_ERROR_THRESHOLD = 0.05
NUMERICAL_DIFFERENTIATION_ERROR_THRESHOLD = 1e-4
SOFTMAX_SUM_ERROR_THRESHOLD = 1e-6


def _relative_error(observed: float, reference: float) -> float:
    """기준값 대비 상대오차를 계산하고, 영 기준값도 안전하게 처리합니다."""
    if reference == 0.0:
        return 0.0 if observed == 0.0 else float("inf")
    return abs(observed - reference) / abs(reference)


def print_submission_criteria() -> None:
    """면적비, 고유값, 수치미분, Softmax의 기준 대비 오차를 출력합니다."""
    area_matrix = np.array([[3.0, 0.0], [0.0, 0.5]])
    determinant, area_ratio, area_relative_error = area_scale_error(
        area_matrix,
        unit_circle(),
    )

    eigen_matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    eigen_comparison = power_iteration_comparison(
        eigen_matrix,
        initial_vector=np.array([1.0, 1.0]),
    )
    power_eigenvalue = float(eigen_comparison["power_eigenvalue"])
    reference_eigenvalue = float(eigen_comparison["reference_eigenvalue"])
    eigenvalue_relative_error = _relative_error(
        power_eigenvalue,
        reference_eigenvalue,
    )

    differentiation_x = 3.0
    numerical_derivative = central_difference(
        lambda value: value**2,
        differentiation_x,
    )
    analytical_derivative = 2.0 * differentiation_x
    differentiation_absolute_error = abs(
        numerical_derivative - analytical_derivative
    )

    logits = np.array([1000.0, 1001.0, 1002.0])
    probabilities = softmax(logits)
    probability_sum = float(probabilities.sum())
    softmax_sum_difference = abs(probability_sum - 1.0)

    print("Submission criteria verification")
    print()
    print(f"[1] Area ratio vs |det(A)| (relative error <= {AREA_ERROR_THRESHOLD:.2%})")
    print(f"matrix               = {area_matrix.tolist()}")
    print(f"det(A)               = {determinant:.12f}")
    print(f"|det(A)|             = {abs(determinant):.12f}")
    print(f"measured area ratio  = {area_ratio:.12f}")
    print(f"relative error       = {area_relative_error:.6%}")
    print()
    print(
        "[2] Power Iteration vs np.linalg.eig "
        f"(relative error <= {EIGENVALUE_ERROR_THRESHOLD:.2%})"
    )
    print(f"matrix               = {eigen_matrix.tolist()}")
    print(f"power eigenvalue     = {power_eigenvalue:.12f}")
    print(f"np.linalg.eig value  = {reference_eigenvalue:.12f}")
    print(
        "eigenvalue abs error = "
        f"{float(eigen_comparison['eigenvalue_absolute_error']):.3e}"
    )
    print(f"relative error       = {eigenvalue_relative_error:.6%}")
    print(f"iterations           = {eigen_comparison['iterations']}")
    print()
    print(
        "[3] Numerical differentiation vs analytical derivative "
        f"(absolute error <= {NUMERICAL_DIFFERENTIATION_ERROR_THRESHOLD:.1e})"
    )
    print("function              = f(x) = x^2, x = 3")
    print(f"numerical derivative  = {numerical_derivative:.12f}")
    print(f"analytical derivative = {analytical_derivative:.12f}")
    print(f"absolute error         = {differentiation_absolute_error:.3e}")
    print()
    print(
        "[4] Softmax output sum "
        f"(difference from 1 <= {SOFTMAX_SUM_ERROR_THRESHOLD:.1e})"
    )
    print(f"logits                 = {logits.tolist()}")
    print(f"probabilities          = {probabilities}")
    print(f"probability sum        = {probability_sum:.15f}")
    print(f"difference from 1      = {softmax_sum_difference:.3e}")


if __name__ == "__main__":
    print_submission_criteria()
