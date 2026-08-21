# Model Learning Mechanics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete required NumPy mathematics lab with reproducible notebooks, figures, tests, and documentation.

**Architecture:** Numerical and plotting behavior lives in four focused `src` modules. Thin notebooks explain derivations and verify fixed examples, while one script regenerates every submitted PNG from deterministic inputs.

**Tech Stack:** Python 3.8+, NumPy, Matplotlib, Seaborn, pytest, Jupyter/nbconvert

**Spec:** `docs/superpowers/specs/2026-08-21-model-learning-mechanics-design.md`

## Global Constraints

- Implement required mission work only; exclude Adam, Newton's method, and information-theory bonuses.
- Use NumPy for computation and Matplotlib/Seaborn only for visualization.
- Do not use PyTorch, TensorFlow, JAX, scikit-learn PCA, or scikit-learn optimizers.
- Use `np.linalg.eig` only in Power Iteration verification tests.
- Set `np.random.seed(42)` in reproducible examples.
- Give every public function and class a docstring and include formulas in code comments/docstrings or notebook Markdown.
- Save final figures under `outputs/` and do not commit `docs/private/`.
- Use test-first red-green-refactor cycles for Python behavior.
- Commit each task with author name `shannonlee-dev` and the Conventional Commit subject shown below.

---

### Task 1: Project Foundation and Linear Algebra

**Files:**
- Create: `src/__init__.py`
- Create: `src/linear_algebra.py`
- Create: `tests/test_linear_algebra.py`
- Create: `requirements.txt`

**Interfaces:**
- Produces: `unit_circle(num_points: int = 361) -> np.ndarray`
- Produces: `rotation_matrix(theta: float) -> np.ndarray`
- Produces: `scaling_matrix(sx: float, sy: float) -> np.ndarray`
- Produces: `shear_matrix(k: float) -> np.ndarray`
- Produces: `transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray`
- Produces: `polygon_area(points: np.ndarray) -> float`
- Produces: `area_scale_error(matrix: np.ndarray, points: np.ndarray) -> Tuple[float, float, float]`
- Produces: `power_iteration(matrix, initial_vector=None, max_iterations=1000, tolerance=1e-9) -> Tuple[float, np.ndarray, int]`
- Produces: `compress_image_svd(image: np.ndarray, k: int) -> np.ndarray`
- Produces: `plot_matrix_transform(matrix, title, points=None, ax=None) -> Tuple[Figure, Axes]`
- Produces: `plot_svd_reconstructions(image, ranks=(10, 50, 100)) -> Tuple[Figure, np.ndarray]`

- [ ] **Step 1: Add the environment manifest and package marker**

Create `requirements.txt` with Python-3.8-compatible minimums: `numpy>=1.21`, `matplotlib>=3.4`, `seaborn>=0.11`, `pytest>=7.0`, `jupyter>=1.0`, and `nbconvert>=6.0`. Add a package docstring to `src/__init__.py`.

- [ ] **Step 2: Write failing transformation and area tests**

```python
def test_standard_transforms_have_expected_geometry():
    point = np.array([[1.0], [0.0]])
    np.testing.assert_allclose(rotation_matrix(np.pi / 2) @ point, [[0.0], [1.0]], atol=1e-12)
    np.testing.assert_allclose(scaling_matrix(2.0, 0.5) @ point, [[2.0], [0.0]])
    np.testing.assert_allclose(shear_matrix(1.5) @ np.array([[0.0], [1.0]]), [[1.5], [1.0]])

@pytest.mark.parametrize("matrix", [rotation_matrix(0.4), scaling_matrix(2, 0.5), shear_matrix(1.2)])
def test_area_ratio_matches_absolute_determinant_within_one_percent(matrix):
    determinant, ratio, error = area_scale_error(matrix, unit_circle())
    assert error <= 0.01
    assert ratio == pytest.approx(abs(determinant), rel=0.01)
```

- [ ] **Step 3: Run the transformation tests and verify the missing-module failure**

Run: `python3 -m pytest tests/test_linear_algebra.py -k 'standard_transforms or area_ratio' -v`

Expected: collection fails because `src.linear_algebra` does not exist.

- [ ] **Step 4: Implement transformation, area, and overlay plotting behavior**

Use the matrices `[[cos θ,-sin θ],[sin θ,cos θ]]`, `diag(sx,sy)`, and `[[1,k],[0,1]]`. Represent point sets as `(2,n)`, close polygons internally with `np.roll`, compute the shoelace area, and reject wrong shapes or fewer than three points. Plot original and transformed points on equal-aspect axes.

- [ ] **Step 5: Run the focused transformation tests until green**

Run: `python3 -m pytest tests/test_linear_algebra.py -k 'standard_transforms or area_ratio' -v`

Expected: all selected tests pass.

- [ ] **Step 6: Write failing Power Iteration tests**

```python
def test_power_iteration_matches_dominant_numpy_eigenpair():
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    value, vector, iterations = power_iteration(matrix, initial_vector=np.array([1.0, 1.0]))
    reference_values, reference_vectors = np.linalg.eig(matrix)
    index = np.argmax(np.abs(reference_values))
    assert value == pytest.approx(reference_values[index], rel=0.05)
    assert abs(vector @ reference_vectors[:, index]) == pytest.approx(1.0, abs=1e-5)
    assert 1 <= iterations <= 1000

def test_power_iteration_rejects_zero_initial_vector():
    with pytest.raises(ValueError, match="initial vector"):
        power_iteration(np.eye(2), initial_vector=np.zeros(2))
```

- [ ] **Step 7: Run the Power Iteration tests and verify they fail for missing behavior**

Run: `python3 -m pytest tests/test_linear_algebra.py -k power_iteration -v`

Expected: tests fail because `power_iteration` is absent.

- [ ] **Step 8: Implement normalized Power Iteration**

Validate a finite, non-empty square matrix, a matching nonzero vector, positive tolerance, and positive iteration limit. Normalize `A @ v`, stop when `min(||v_next-v||, ||v_next+v||) <= tolerance`, and return the Rayleigh quotient `v.T @ A @ v`, normalized vector, and iteration count. Raise `RuntimeError` if convergence is not reached.

- [ ] **Step 9: Run the Power Iteration tests until green**

Run: `python3 -m pytest tests/test_linear_algebra.py -k power_iteration -v`

Expected: all selected tests pass without `np.linalg.eig` in production code.

- [ ] **Step 10: Write failing SVD reconstruction tests**

```python
def test_svd_rank_reconstruction_error_decreases_with_k():
    image = np.random.default_rng(42).random((128, 128))
    error_10 = np.linalg.norm(image - compress_image_svd(image, 10))
    error_50 = np.linalg.norm(image - compress_image_svd(image, 50))
    error_100 = np.linalg.norm(image - compress_image_svd(image, 100))
    assert error_100 < error_50 < error_10

@pytest.mark.parametrize("k", [0, 129])
def test_svd_rejects_rank_outside_image_dimensions(k):
    with pytest.raises(ValueError, match="rank"):
        compress_image_svd(np.ones((128, 128)), k)
```

- [ ] **Step 11: Run the SVD tests and verify they fail for missing behavior**

Run: `python3 -m pytest tests/test_linear_algebra.py -k svd -v`

Expected: tests fail because `compress_image_svd` is absent.

- [ ] **Step 12: Implement truncated SVD and comparison plotting**

Compute `U, singular_values, Vt = np.linalg.svd(image, full_matrices=False)` and reconstruct `(U[:, :k] * singular_values[:k]) @ Vt[:k, :]`. Reject non-2D, empty, non-finite arrays and invalid ranks. Plot original plus each requested rank using a grayscale colormap.

- [ ] **Step 13: Run all linear-algebra tests**

Run: `MPLBACKEND=Agg python3 -m pytest tests/test_linear_algebra.py -v`

Expected: all tests pass with no warnings.

- [ ] **Step 14: Commit the feature**

```bash
git add requirements.txt src/__init__.py src/linear_algebra.py tests/test_linear_algebra.py
git -c user.name=shannonlee-dev commit -m "feat(linear-algebra): add transformation and decomposition lab"
```

### Task 2: Numerical Calculus and Gradient Visualization

**Files:**
- Create: `src/calculus.py`
- Create: `tests/test_calculus.py`

**Interfaces:**
- Produces: `central_difference(function, x: float, h: float = 1e-5) -> float`
- Produces: `numerical_gradient(function, point: np.ndarray, h: float = 1e-5) -> np.ndarray`
- Produces: `quadratic_bowl(x, y) -> Union[np.ndarray, float]`
- Produces: `quadratic_bowl_gradient(point: np.ndarray) -> np.ndarray`
- Produces: `plot_gradient_field(function=quadratic_bowl, gradient=quadratic_bowl_gradient, sample_points=None) -> Tuple[Figure, Axes]`

- [ ] **Step 1: Write failing numerical differentiation tests**

```python
def test_central_difference_matches_square_derivative_at_three():
    assert central_difference(lambda x: x**2, 3.0) == pytest.approx(6.0, abs=1e-4)

def test_numerical_gradient_matches_quadratic_gradient():
    point = np.array([2.0, -3.0])
    np.testing.assert_allclose(
        numerical_gradient(lambda p: p[0] ** 2 + p[1] ** 2, point),
        [4.0, -6.0],
        atol=1e-4,
    )

def test_gradient_is_perpendicular_to_contour_tangent():
    gradient = quadratic_bowl_gradient(np.array([2.0, 1.0]))
    tangent = np.array([-gradient[1], gradient[0]])
    assert gradient @ tangent == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 2: Run calculus tests and verify the missing-module failure**

Run: `python3 -m pytest tests/test_calculus.py -v`

Expected: collection fails because `src.calculus` does not exist.

- [ ] **Step 3: Implement central differences and coordinate-wise gradients**

Apply `(f(x+h)-f(x-h))/(2h)` to scalars and to one coordinate at a time for a one-dimensional point vector. Reject non-positive `h`, empty/non-finite points, and non-scalar function results.

- [ ] **Step 4: Implement quadratic contours and normalized gradient arrows**

Create a mesh with `np.meshgrid`, draw `x**2+y**2` contours, and use `Axes.quiver` for gradients at fixed non-origin sample points. Return the figure and axes and preserve equal axis scaling.

- [ ] **Step 5: Run calculus tests until green**

Run: `MPLBACKEND=Agg python3 -m pytest tests/test_calculus.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the feature**

```bash
git add src/calculus.py tests/test_calculus.py
git -c user.name=shannonlee-dev commit -m "feat(calculus): add numerical gradient visualization"
```

### Task 3: Backpropagation Derivation

**Files:**
- Create: `notebooks/backprop_derivation.ipynb`

**Interfaces:**
- Consumes: NumPy from `requirements.txt`
- Produces: an executable notebook with fixed forward/backward checkpoints and assertions

- [ ] **Step 1: Hand-calculate and record literal expected checkpoints**

Use these hand-derived four-decimal literals: `z1=[0.1000,0.3000]`, `a1=[0.5250,0.5744]`, `z2=0.6072`, `y_pred=0.6473`, `dL/dy_pred=-1.5449`, `dL/dz2=-0.3527`, `dL/dW2=[-0.1852,-0.2026]`, `dL/da1=[-0.1764,-0.2116]`, `dL/dz1=[-0.0440,-0.0517]`, and `dL/dW1=[[-0.0440,0],[-0.0517,0]]`. These literals must not be generated from the implementation inside the comparison assertion.

- [ ] **Step 2: Create the notebook derivation and executable verification**

Add Markdown cells for the network shapes, forward equations, BCE loss, sigmoid derivative, and every required chain-rule derivative. Add code cells that compute the same values with NumPy, print a table of names/shapes/values, and use `np.testing.assert_array_almost_equal(..., decimal=4)` against the recorded literals.

- [ ] **Step 3: Execute the notebook from a clean kernel**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/backprop_derivation.ipynb --ExecutePreprocessor.timeout=120`

Expected: execution succeeds and all four-decimal assertions pass.

- [ ] **Step 4: Inspect notebook output for all required checkpoints**

Run: `jupyter nbconvert --to markdown --stdout notebooks/backprop_derivation.ipynb`

Expected: output contains `z1`, `a1`, `z2`, `y_pred`, `dL/dy_pred`, `dL/dz2`, `dL/dW2`, `dL/da1`, `dL/dz1`, and `dL/dW1`, with shapes and verification success.

- [ ] **Step 5: Commit the notebook**

```bash
git add notebooks/backprop_derivation.ipynb
git -c user.name=shannonlee-dev commit -m "docs(backprop): derive two-layer network gradients"
```

### Task 4: Gradient Descent and Momentum

**Files:**
- Create: `src/optimizer.py`
- Create: `tests/test_optimizer.py`

**Interfaces:**
- Produces: `VanillaGD(learning_rate: float)` with `update(parameters, gradient) -> np.ndarray`
- Produces: `Momentum(learning_rate: float, beta: float = 0.9)` with `update(parameters, gradient) -> np.ndarray`
- Produces: `optimize(gradient_function, initial_point, optimizer, steps: int) -> np.ndarray`
- Produces: `sphere(point) -> float`, `sphere_gradient(point) -> np.ndarray`
- Produces: `elliptical(point) -> float`, `elliptical_gradient(point) -> np.ndarray`
- Produces: `plot_optimization_paths(objective, paths, x_limits, y_limits, title) -> Tuple[Figure, Axes]`

- [ ] **Step 1: Write failing optimizer update and validation tests**

```python
def test_vanilla_gd_applies_gradient_step():
    updated = VanillaGD(0.1).update(np.array([2.0, -1.0]), np.array([4.0, -2.0]))
    np.testing.assert_allclose(updated, [1.6, -0.8])

def test_momentum_accumulates_velocity():
    optimizer = Momentum(learning_rate=0.1, beta=0.9)
    first = optimizer.update(np.array([1.0]), np.array([2.0]))
    second = optimizer.update(first, np.array([2.0]))
    np.testing.assert_allclose(first, [0.8])
    np.testing.assert_allclose(second, [0.42])

@pytest.mark.parametrize("factory", [lambda: VanillaGD(0), lambda: Momentum(0.1, beta=1)])
def test_optimizers_reject_invalid_hyperparameters(factory):
    with pytest.raises(ValueError):
        factory()
```

- [ ] **Step 2: Run update tests and verify the missing-module failure**

Run: `python3 -m pytest tests/test_optimizer.py -k 'gd or momentum or hyperparameters' -v`

Expected: collection fails because `src.optimizer` does not exist.

- [ ] **Step 3: Implement optimizer state and shape validation**

Vanilla uses `parameters - learning_rate*gradient`. Momentum lazily initializes a zero velocity matching the first parameter vector, applies `velocity = beta*velocity + gradient`, and updates parameters. Reject empty/non-finite vectors and mismatched shapes.

- [ ] **Step 4: Run update tests until green**

Run: `python3 -m pytest tests/test_optimizer.py -k 'gd or momentum or hyperparameters' -v`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing convergence, divergence, and comparison tests**

```python
def test_vanilla_gd_reaches_radius_point_one_after_100_steps():
    path = optimize(sphere_gradient, [5.0, 5.0], VanillaGD(0.1), steps=100)
    assert path.shape == (101, 2)
    assert np.linalg.norm(path[-1]) <= 0.1

def test_learning_rate_one_point_one_diverges_on_sphere():
    path = optimize(sphere_gradient, [5.0, 5.0], VanillaGD(1.1), steps=20)
    assert np.linalg.norm(path[-1]) > np.linalg.norm(path[0])

def test_momentum_advances_faster_on_elliptical_bowl():
    vanilla = optimize(elliptical_gradient, [5.0, 5.0], VanillaGD(0.01), steps=60)
    momentum = optimize(elliptical_gradient, [5.0, 5.0], Momentum(0.01, 0.9), steps=60)
    assert elliptical(momentum[-1]) < elliptical(vanilla[-1])
```

- [ ] **Step 6: Run path tests and verify they fail for missing behavior**

Run: `python3 -m pytest tests/test_optimizer.py -k 'reaches or diverges or faster' -v`

Expected: tests fail because runner/objective behavior is absent.

- [ ] **Step 7: Implement objectives, path collection, and contour overlays**

Store the initial point at path index zero, then append exactly one point per update. Implement `sphere=x^2+y^2`, gradient `[2x,2y]`, `elliptical=x^2+10y^2`, and gradient `[2x,20y]`. Plot each named path with dashed lines, markers, start/end emphasis, and equal scaling.

- [ ] **Step 8: Run all optimizer tests**

Run: `MPLBACKEND=Agg python3 -m pytest tests/test_optimizer.py -v`

Expected: all tests pass.

- [ ] **Step 9: Commit the feature**

```bash
git add src/optimizer.py tests/test_optimizer.py
git -c user.name=shannonlee-dev commit -m "feat(optimizer): compare gradient descent and momentum"
```

### Task 5: Probability Utilities and Likelihood Derivations

**Files:**
- Create: `src/probability.py`
- Create: `tests/test_probability.py`
- Create: `notebooks/probability_loss.ipynb`

**Interfaces:**
- Produces: `normal_pdf(x, mean: float = 0.0, std: float = 1.0) -> np.ndarray`
- Produces: `bernoulli_pmf(outcomes, probability: float) -> np.ndarray`
- Produces: `softmax(logits: np.ndarray) -> np.ndarray`
- Produces: `plot_normal_distributions() -> Tuple[Figure, Axes]`
- Produces: `plot_bernoulli_distributions() -> Tuple[Figure, Axes]`

- [ ] **Step 1: Write failing probability tests**

```python
def test_standard_normal_pdf_at_zero():
    assert normal_pdf(np.array([0.0]))[0] == pytest.approx(1 / np.sqrt(2 * np.pi))

def test_bernoulli_pmf_returns_failure_and_success_probabilities():
    np.testing.assert_allclose(bernoulli_pmf(np.array([0, 1]), 0.3), [0.7, 0.3])

def test_softmax_is_stable_and_normalized():
    probabilities = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.isfinite(probabilities).all()
    assert abs(probabilities.sum() - 1.0) <= 1e-6
    assert np.argmax(probabilities) == 2

@pytest.mark.parametrize("call", [lambda: normal_pdf([0], std=0), lambda: bernoulli_pmf([0], -0.1), lambda: softmax([])])
def test_probability_functions_reject_invalid_parameters(call):
    with pytest.raises(ValueError):
        call()
```

- [ ] **Step 2: Run probability tests and verify the missing-module failure**

Run: `python3 -m pytest tests/test_probability.py -v`

Expected: collection fails because `src.probability` does not exist.

- [ ] **Step 3: Implement distributions, stable Softmax, and plots**

Use the normal density `exp(-0.5*((x-mean)/std)^2)/(std*sqrt(2π))`, Bernoulli mass `p^x*(1-p)^(1-x)` for outcomes restricted to 0 or 1, and `exp(logits-max(logits))/sum(exp(...))`. Plot both requested parameterizations on shared axes.

- [ ] **Step 4: Run probability tests until green**

Run: `MPLBACKEND=Agg python3 -m pytest tests/test_probability.py -v`

Expected: all tests pass.

- [ ] **Step 5: Create the probability and loss notebook**

Add executed plots for `N(0,1)`, `N(2,0.5)`, `B(0.3)`, and `B(0.7)` using the module functions. Add a Softmax normalization assertion. In Markdown, derive Gaussian log-likelihood to `argmin Σ(y_i-f(x_i))²`, Bernoulli negative log-likelihood to BCE, and categorical negative log-likelihood to multiclass cross-entropy, explicitly marking removed constants.

- [ ] **Step 6: Execute and inspect the probability notebook**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/probability_loss.ipynb --ExecutePreprocessor.timeout=120`

Expected: execution succeeds, plots render, and the Softmax sum assertion passes.

- [ ] **Step 7: Commit the feature**

```bash
git add src/probability.py tests/test_probability.py notebooks/probability_loss.ipynb
git -c user.name=shannonlee-dev commit -m "feat(probability): connect distributions and losses"
```

### Task 6: Reproducible Artifacts and Project Documentation

**Files:**
- Create: `scripts/generate_outputs.py`
- Create: `tests/test_generate_outputs.py`
- Create: `outputs/*.png`
- Create: `README.md`

**Interfaces:**
- Consumes: all public plotting and optimization interfaces from Tasks 1, 2, 4, and 5
- Produces: `generate_all_outputs(output_directory: Path) -> List[Path]`
- Produces: a repository-root CLI command `python3 scripts/generate_outputs.py`

- [ ] **Step 1: Write a failing end-to-end artifact test**

```python
EXPECTED_OUTPUTS = {
    "rotation_transform.png", "scaling_transform.png", "shear_transform.png",
    "svd_compression.png", "gradient_field.png", "gd_convergence_divergence.png",
    "optimizer_comparison.png", "elliptical_optimizer_comparison.png",
    "normal_distributions.png", "bernoulli_distributions.png",
}

def test_generate_all_outputs_creates_nonempty_pngs(tmp_path):
    paths = generate_all_outputs(tmp_path)
    assert {path.name for path in paths} == EXPECTED_OUTPUTS
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in paths)
    assert all(path.stat().st_size > 1_000 for path in paths)
```

- [ ] **Step 2: Run the artifact test and verify the missing-module failure**

Run: `MPLBACKEND=Agg python3 -m pytest tests/test_generate_outputs.py -v`

Expected: collection fails because `scripts.generate_outputs` does not exist.

- [ ] **Step 3: Implement deterministic artifact generation**

Fix the seed to 42, create a deterministic 128×128 grayscale sample from smooth gradients plus geometric masks, call each plotting function, save figures with `dpi=150` and `bbox_inches="tight"`, close every figure, and return the ten paths in filename order. Add a `__main__` block targeting repository-root `outputs/`.

- [ ] **Step 4: Run the artifact test until green**

Run: `MPLBACKEND=Agg python3 -m pytest tests/test_generate_outputs.py -v`

Expected: all ten PNGs are valid and non-empty.

- [ ] **Step 5: Generate the committed outputs**

Run: `MPLBACKEND=Agg python3 scripts/generate_outputs.py`

Expected: ten named PNG files are written under `outputs/`.

- [ ] **Step 6: Write the README**

Document the project goal, formulas demonstrated, tree, Python 3.8+ setup, `pip install -r requirements.txt`, pytest command, output-generation command, notebook execution commands, expected checkpoints (`f'(3)≈6`, eigenvalue `≈4.618`, Softmax sum `≈1`, GD radius `<0.1`), output gallery, and why `lr=1.1` is used for genuine divergence.

- [ ] **Step 7: Run aggregate verification**

Run:

```bash
MPLBACKEND=Agg python3 -m pytest -v
jupyter nbconvert --to notebook --execute --inplace notebooks/backprop_derivation.ipynb --ExecutePreprocessor.timeout=120
jupyter nbconvert --to notebook --execute --inplace notebooks/probability_loss.ipynb --ExecutePreprocessor.timeout=120
MPLBACKEND=Agg python3 scripts/generate_outputs.py
MPLBACKEND=Agg python3 -m pytest -v
```

Expected: every command exits zero, both notebooks retain executed outputs, all ten figures regenerate, and the final test run is clean.

- [ ] **Step 8: Commit documentation and artifacts**

```bash
git add README.md scripts/generate_outputs.py tests/test_generate_outputs.py outputs
git -c user.name=shannonlee-dev commit -m "docs: add reproducible results and usage guide"
```

### Task 7: Final Repository Audit

**Files:**
- Inspect: all tracked files
- Leave untracked: `docs/private/`

**Interfaces:**
- Consumes: all task deliverables
- Produces: verified feature commits and a clean submission diff except private source documents

- [ ] **Step 1: Check requirement coverage and forbidden dependencies**

Run: `rg -n 'torch|tensorflow|jax|sklearn|np\.linalg\.eig' src notebooks scripts README.md tests`

Expected: no forbidden libraries; `np.linalg.eig` appears only in `tests/test_linear_algebra.py`.

- [ ] **Step 2: Inspect all commits and remaining changes**

Run:

```bash
git log --oneline --decorate
git status --short
git diff --check
git show --check HEAD
```

Expected: feature commits use Conventional Commits, no tracked changes remain, and only `docs/private/` is untracked.

- [ ] **Step 3: Verify commit authors**

Run: `git log --format='%h %an <%ae> %s'`

Expected: every new commit author name is exactly `shannonlee-dev`.
