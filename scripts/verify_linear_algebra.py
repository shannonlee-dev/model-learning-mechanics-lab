"""선형대수 제출 기준의 수치 검증 결과를 콘솔에 출력합니다."""

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

from src.linear_algebra import area_scale_error, power_iteration_comparison, unit_circle


AREA_ERROR_THRESHOLD = 0.01
EIGENVALUE_ERROR_THRESHOLD = 0.05


def _relative_error(observed: float, reference: float) -> float:
    """기준값 대비 상대오차를 계산하고, 영 기준값도 안전하게 처리합니다."""
    if reference == 0.0:
        return 0.0 if observed == 0.0 else float("inf")
    return abs(observed - reference) / abs(reference)


def print_linear_algebra_verification() -> None:
    """면적비와 주 고유값의 기준 대비 오차·통과 여부를 출력합니다."""
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

    print("Linear algebra verification")
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


if __name__ == "__main__":
    print_linear_algebra_verification()
