"""결정론적 입력으로 제출한 모든 시각화를 다시 생성합니다."""

import os
import sys
import tempfile
from pathlib import Path
from typing import List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SVD_SOURCE_IMAGE = REPOSITORY_ROOT / "assets" / "images" / "svd_source.png"
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "model-learning-mechanics-matplotlib"),
)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.calculus import plot_gradient_field  # noqa: E402
from src.linear_algebra import (  # noqa: E402
    area_scale_error,
    plot_matrix_transform,
    plot_svd_reconstructions,
    power_iteration_comparison,
    rotation_matrix,
    scaling_matrix,
    shear_matrix,
    unit_circle,
)
from src.optimizer import (  # noqa: E402
    Momentum,
    VanillaGD,
    elliptical,
    elliptical_gradient,
    optimize,
    plot_optimization_paths,
    sphere,
    sphere_gradient,
)
from src.probability import (  # noqa: E402
    plot_bernoulli_distributions,
    plot_normal_distributions,
)


def _save_figure(figure, path: Path) -> Path:
    """Figure 하나를 PNG로 저장하고 닫은 뒤 저장 경로를 반환합니다."""
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def load_svd_source_image() -> np.ndarray:
    """제출한 SVD 원본 이미지를 정규화한 grayscale 배열로 불러옵니다."""
    image = np.asarray(plt.imread(SVD_SOURCE_IMAGE), dtype=float)
    if image.ndim == 2:
        grayscale = image
    elif image.ndim == 3 and image.shape[2] >= 3:
        grayscale = image[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    else:
        raise ValueError("SVD source image must be grayscale or RGB")
    if grayscale.max() > 1.0:
        grayscale /= 255.0
    return np.clip(grayscale, 0.0, 1.0)


def generate_all_outputs(output_directory: Path) -> List[Path]:
    """실습에 필요한 결정론적 PNG 결과 열 개를 생성하고 저장합니다."""
    np.random.seed(42)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    generated = []

    transforms = [
        ("rotation_transform.png", rotation_matrix(np.pi / 4), "Rotation R(45°)"),
        ("scaling_transform.png", scaling_matrix(2.0, 0.5), "Scaling S(2, 0.5)"),
        ("shear_transform.png", shear_matrix(1.0), "Shear Sh(1.0)"),
    ]
    circle = unit_circle()
    for filename, matrix, title in transforms:
        determinant, ratio, error = area_scale_error(matrix, circle)
        full_title = (
            f"{title}\n"
            f"det={determinant:.3f}, area ratio={ratio:.3f}, error={error:.2%}"
        )
        figure, _ = plot_matrix_transform(matrix, full_title, points=circle)
        generated.append(_save_figure(figure, destination / filename))

    svd_figure, _ = plot_svd_reconstructions(load_svd_source_image())
    generated.append(_save_figure(svd_figure, destination / "svd_compression.png"))

    gradient_figure, _ = plot_gradient_field()
    generated.append(_save_figure(gradient_figure, destination / "gradient_field.png"))

    convergent = optimize(sphere_gradient, [5.0, 5.0], VanillaGD(0.1), steps=100)
    divergent = optimize(sphere_gradient, [5.0, 5.0], VanillaGD(1.1), steps=12)
    convergence_figure, _ = plot_optimization_paths(
        sphere,
        {"lr=0.1 (convergent)": convergent, "lr=1.1 (divergent)": divergent},
        x_limits=(-50.0, 50.0),
        y_limits=(-50.0, 50.0),
        title="Vanilla GD: learning-rate stability",
    )
    generated.append(
        _save_figure(convergence_figure, destination / "gd_convergence_divergence.png")
    )

    vanilla_sphere = optimize(
        sphere_gradient,
        [5.0, 5.0],
        VanillaGD(0.1),
        steps=100,
    )
    momentum_sphere = optimize(
        sphere_gradient,
        [5.0, 5.0],
        Momentum(0.1, beta=0.9),
        steps=100,
    )
    optimizer_figure, _ = plot_optimization_paths(
        sphere,
        {"Vanilla GD": vanilla_sphere, "Momentum β=0.9": momentum_sphere},
        x_limits=(-6.0, 6.0),
        y_limits=(-6.0, 6.0),
        title="Optimizer paths on f(x, y) = x² + y²",
    )
    generated.append(
        _save_figure(optimizer_figure, destination / "optimizer_comparison.png")
    )

    vanilla_elliptical = optimize(
        elliptical_gradient,
        [5.0, 5.0],
        VanillaGD(0.01),
        steps=60,
    )
    momentum_elliptical = optimize(
        elliptical_gradient,
        [5.0, 5.0],
        Momentum(0.01, beta=0.9),
        steps=60,
    )
    elliptical_figure, _ = plot_optimization_paths(
        elliptical,
        {"Vanilla GD": vanilla_elliptical, "Momentum β=0.9": momentum_elliptical},
        x_limits=(-6.0, 6.0),
        y_limits=(-6.0, 6.0),
        title="Optimizer paths on f(x, y) = x² + 10y²",
    )
    generated.append(
        _save_figure(
            elliptical_figure,
            destination / "elliptical_optimizer_comparison.png",
        )
    )

    normal_figure, _ = plot_normal_distributions()
    generated.append(_save_figure(normal_figure, destination / "normal_distributions.png"))
    bernoulli_figure, _ = plot_bernoulli_distributions()
    generated.append(
        _save_figure(bernoulli_figure, destination / "bernoulli_distributions.png")
    )
    return generated


def print_power_iteration_comparison() -> None:
    """재현 가능한 Power Iteration과 NumPy eig 비교 결과를 출력합니다."""
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    comparison = power_iteration_comparison(
        matrix,
        initial_vector=np.array([1.0, 1.0]),
    )
    print("Power Iteration vs np.linalg.eig")
    print(f"matrix = {matrix.tolist()}")
    print(f"power eigenvalue     = {comparison['power_eigenvalue']:.12f}")
    print(f"numpy eig eigenvalue = {comparison['reference_eigenvalue']:.12f}")
    print(
        "eigenvalue abs error = "
        f"{comparison['eigenvalue_absolute_error']:.3e}"
    )
    print(f"power eigenvector     = {comparison['power_eigenvector']}")
    print(f"numpy eig eigenvector = {comparison['reference_eigenvector']}")
    print(f"eigenvector alignment = {comparison['eigenvector_alignment']:.12f}")
    print(f"eigenvector L2 error  = {comparison['eigenvector_l2_error']:.3e}")
    print(f"iterations            = {comparison['iterations']}")


if __name__ == "__main__":
    result_paths = generate_all_outputs(REPOSITORY_ROOT / "outputs")
    for result_path in result_paths:
        print(result_path.relative_to(REPOSITORY_ROOT))
    print_power_iteration_comparison()
