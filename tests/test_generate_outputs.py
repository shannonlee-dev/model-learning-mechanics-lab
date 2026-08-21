"""End-to-end tests for reproducible result artifacts."""

from scripts.generate_outputs import generate_all_outputs


EXPECTED_OUTPUTS = {
    "rotation_transform.png",
    "scaling_transform.png",
    "shear_transform.png",
    "svd_compression.png",
    "gradient_field.png",
    "gd_convergence_divergence.png",
    "optimizer_comparison.png",
    "elliptical_optimizer_comparison.png",
    "normal_distributions.png",
    "bernoulli_distributions.png",
}


def test_generate_all_outputs_creates_nonempty_pngs(tmp_path):
    """Skipping or corrupting any figure must break the artifact contract."""
    paths = generate_all_outputs(tmp_path)
    assert {path.name for path in paths} == EXPECTED_OUTPUTS
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in paths)
    assert all(path.stat().st_size > 1_000 for path in paths)
