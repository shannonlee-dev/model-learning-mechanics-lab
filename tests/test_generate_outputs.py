"""재현 가능한 결과 산출물에 대한 엔드투엔드 테스트입니다."""

import numpy as np
import matplotlib.pyplot as plt

from scripts.generate_outputs import SVD_SOURCE_IMAGE, generate_all_outputs, load_svd_source_image


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
    """Figure를 누락하거나 훼손하면 산출물 조건이 깨져야 합니다."""
    paths = generate_all_outputs(tmp_path)
    assert {path.name for path in paths} == EXPECTED_OUTPUTS
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in paths)
    assert all(path.stat().st_size > 1_000 for path in paths)


def test_svd_source_image_loads_the_uploaded_image_as_grayscale():
    """업로드한 원본을 합성 데이터로 바꾸면 이 조건이 깨져야 합니다."""
    image = load_svd_source_image()

    assert SVD_SOURCE_IMAGE.is_file()
    assert image.ndim == 2
    assert min(image.shape) >= 100
    assert np.isfinite(image).all()
    assert 0.0 <= image.min() <= image.max() <= 1.0


def test_optimizer_outputs_use_wide_diagnostic_layouts(tmp_path):
    """경로와 수렴 지표를 함께 보여주는 다중 패널 구성이 사라지면 실패해야 합니다."""
    generated = {path.name: path for path in generate_all_outputs(tmp_path)}

    for filename in (
        "gd_convergence_divergence.png",
        "optimizer_comparison.png",
        "elliptical_optimizer_comparison.png",
    ):
        image = plt.imread(generated[filename])
        height, width = image.shape[:2]
        assert width / height >= 1.65
