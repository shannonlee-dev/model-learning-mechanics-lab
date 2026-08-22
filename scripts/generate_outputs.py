"""Regenerate every submitted visualization with deterministic inputs."""

import os
import sys
import tempfile
from pathlib import Path
from typing import List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
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
    """Save one figure as a PNG, close it, and return its destination path."""
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _sample_grayscale_image() -> np.ndarray:
    """Return the deterministic grayscale image used for SVD reconstruction."""
    rng = np.random.default_rng(42)
    coordinates = np.linspace(-1.0, 1.0, 128)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    radial = np.exp(-4.0 * (x_grid**2 + y_grid**2))
    waves = 0.5 + 0.5 * np.sin(12.0 * x_grid) * np.cos(10.0 * y_grid)
    square = ((abs(x_grid) < 0.42) & (abs(y_grid) < 0.42)).astype(float)
    noise = rng.normal(0.0, 0.08, size=x_grid.shape)
    return np.clip(0.35 * radial + 0.30 * waves + 0.27 * square + noise, 0.0, 1.0)


def generate_all_outputs(output_directory: Path) -> List[Path]:
    """Generate and save the ten deterministic PNG results required by the lab."""
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

    svd_figure, _ = plot_svd_reconstructions(_sample_grayscale_image())
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
    """Print the reproducible Power Iteration versus NumPy eig comparison."""
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
