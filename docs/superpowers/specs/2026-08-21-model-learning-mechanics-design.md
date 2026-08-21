# Model Learning Mechanics Lab Design

## Goal

Build a reproducible NumPy-based learning lab that demonstrates the mathematical mechanics behind linear transformations, numerical differentiation, backpropagation, gradient-based optimization, and probabilistic loss functions.

The required mission scope is implemented in full. Optional bonus work—Adam, Newton's method, and information-theory utilities—is excluded.

## Constraints

- Support Python 3.8 or newer.
- Use NumPy for mathematical computation and Matplotlib/Seaborn only for visualization.
- Do not use automatic differentiation frameworks or scikit-learn optimization/PCA implementations.
- Set `np.random.seed(42)` for reproducible examples.
- Give every public function and class a docstring.
- Record mathematical formulas in docstrings, comments, or notebook Markdown.
- Save reproducible visualization artifacts as PNG files under `outputs/`.
- Keep `docs/private/` out of commits because it is source material rather than a submission artifact.

## Repository Structure

```text
src/
  __init__.py
  linear_algebra.py
  calculus.py
  optimizer.py
  probability.py
notebooks/
  backprop_derivation.ipynb
  probability_loss.ipynb
scripts/
  generate_outputs.py
tests/
  test_linear_algebra.py
  test_calculus.py
  test_optimizer.py
  test_probability.py
outputs/
README.md
requirements.txt
```

The `src` package owns reusable numerical and plotting behavior. Notebooks contain the explanatory derivations and call those modules instead of duplicating reusable functions. The output script constructs deterministic examples and regenerates all figures in one command. Tests exercise observable numerical behavior rather than inspecting source text.

## Linear Algebra

`src/linear_algebra.py` provides rotation, scaling, and shear matrix constructors and a `plot_matrix_transform` function that overlays the unit circle and transformed points on one axis with equal aspect ratio. A polygon-area helper uses the shoelace formula on a closed point sequence. The reported area ratio is compared with `abs(det(A))`; standard transformations must agree within 1%.

Power Iteration accepts a real square matrix, optional initial vector, tolerance, and positive iteration limit. It normalizes the vector on every iteration, estimates the dominant eigenvalue with the Rayleigh quotient, treats eigenvectors that differ only by sign as equivalent, and returns the eigenvalue, eigenvector, and completed iteration count. The implementation itself does not call `np.linalg.eig`; tests may use it only as the reference and require less than 5% relative eigenvalue error.

SVD compression accepts a non-empty two-dimensional grayscale array and a positive rank `k` no greater than `min(image.shape)`. It reconstructs the array from the first `k` singular triplets. A comparison plot shows the original beside `k=10`, `k=50`, and `k=100` reconstructions using a deterministic 128×128 grayscale image. The source guide suggests images no larger than 64×64, but that cannot represent a rank-100 truncation; the explicit `k=100` result therefore takes precedence.

## Calculus

`src/calculus.py` implements the central-difference formula

\[
f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}
\]

and a coordinate-wise numerical gradient for two-dimensional points. Both reject non-positive step sizes. The scalar example evaluates `f(x)=x^2` at `x=3` and must be within `1e-4` of 6.

The gradient plot samples `f(x,y)=x^2+y^2`, draws its contours, and overlays normalized gradient arrows at selected non-origin points. Tests verify numerical gradients and verify perpendicularity by checking that each gradient is orthogonal to the local contour tangent `(-g_y, g_x)` within floating-point tolerance.

## Backpropagation Notebook

`notebooks/backprop_derivation.ipynb` documents a two-input, two-hidden-unit, one-output sigmoid network with BCE loss. It fixes:

- `x=[1,0]`
- `W1=[[0.1,0.2],[0.3,0.4]]`, `b1=[0,0]`
- `W2=[0.5,0.6]`, `b2=0`
- `y_true=1`, as established by the mission's worked example

Markdown cells derive the forward values `z1`, `a1`, `z2`, and `y_pred`, then derive `dL/dy_pred`, `dL/dz2`, `dL/dW2`, `dL/da1`, `dL/dz1`, and `dL/dW1` through the chain rule. Every operand and intermediate includes its tensor shape. Code cells repeat the calculation with NumPy and compare the handwritten checkpoints to four decimal places.

## Optimization

`src/optimizer.py` defines stateful `VanillaGD` and `Momentum` classes. Each validates a positive learning rate; Momentum also validates `0 <= beta < 1`. Their update methods consume a parameter vector and same-shaped gradient and return the updated vector. Momentum maintains velocity according to `v_t = beta*v_(t-1) + grad` and updates `theta_t = theta_(t-1) - lr*v_t`.

A runner applies either optimizer to an objective gradient and returns the complete parameter path including the initial point. A contour plotting function overlays one or more named paths using dashed lines and markers.

Experiments cover:

- `x^2+y^2`, Vanilla GD, `lr=0.1`, 100 steps, ending within radius 0.1.
- `x^2+y^2`, Vanilla GD, `lr=1.1`, demonstrating a growing path. The mission asks for a divergent `lr>=0.5` example; `1.1` is chosen because the quadratic update factor is `1-2*lr`, whose magnitude exceeds one only when `lr>1`.
- Vanilla GD and Momentum (`beta=0.9`) overlaid under identical initial conditions.
- `x^2+10y^2`, initial point `[5,5]`, 60 steps, and `lr=0.01` for both optimizers. The comparison reports final objective values and shows Momentum's faster progress along the shallow x-axis while both methods remain finite.

## Probability and Losses

`src/probability.py` implements normal PDFs, Bernoulli PMFs, and numerically stable Softmax. Normal standard deviations must be positive, Bernoulli probabilities must lie in `[0,1]`, and Softmax rejects empty input. Softmax subtracts the maximum logit before exponentiation and must sum to 1 within `1e-6`, including for large logits.

`notebooks/probability_loss.ipynb` uses a fixed seed and displays:

- PDFs for `N(0,1)` and `N(2,0.5)` on one figure.
- PMFs for `B(0.3)` and `B(0.7)` on one figure.
- A numerical Softmax normalization check.
- A stepwise derivation showing that Gaussian negative log-likelihood with constant variance reduces to squared-error minimization.
- Stepwise Bernoulli and categorical likelihood derivations showing that negative log-likelihood reduces to binary or multiclass cross-entropy.

## Output Generation and Documentation

`scripts/generate_outputs.py` creates `outputs/` if necessary, fixes the NumPy seed, closes figures after saving, and generates transformation, SVD, gradient, optimizer convergence/divergence, elliptical optimizer comparison, and probability-distribution PNGs. It must work from the repository root without requiring notebook execution.

`README.md` explains the mathematical goals, repository layout, environment setup, test and output-generation commands, and the observed numerical checkpoints. `requirements.txt` sets compatible minimum versions for NumPy, Matplotlib, Seaborn, pytest, and Jupyter while retaining Python 3.8 compatibility.

## Errors and Validation

Public numerical functions fail early with `ValueError` for invalid shapes, empty arrays, invalid ranks, non-positive iteration limits or step sizes, and invalid optimizer hyperparameters. They do not silently reshape or clip invalid input.

Tests follow red-green-refactor cycles and cover the mission's tolerances, realistic boundary failures, optimizer paths, and numerically stable probability results. Before completion, the full test suite, output generation, notebook execution, and a clean rerun of the tests are required. Generated notebooks and PNGs are committed with their corresponding feature so the repository is directly reviewable.

## Commit Boundaries

Implementation is split into independently reviewable Conventional Commits:

1. Project foundation and linear-algebra experiments.
2. Numerical-calculus and gradient visualization.
3. Backpropagation derivation notebook.
4. Gradient-descent and Momentum experiments.
5. Probability utilities and likelihood-loss notebook.
6. Reproducible output generation and project documentation.

Each feature commit includes its implementation, tests, and directly supporting artifacts. Commit author name is `shannonlee-dev`.
