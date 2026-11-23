# VGSP — Variance-Guided Scalar Projection

**A lightweight, scikit-learn-compatible supervised transformer for one-dimensional feature representation.**

VGSP converts numeric multidimensional features into **one CVR-weighted scalar**. It computes a bounded Class-Variance Ratio (CVR) from labelled training data, selects the top-*k* features, and projects them onto a single axis. The project originated in research on lightweight DDoS detection in Software-Defined Networks (SDN), but the transformer can be used with other suitable **binary-classification** datasets.

Unlike PCA, VGSP is **supervised**: `fit(X, y)` needs class labels. The package provides a *feature transformer*, not a pretrained DDoS detector.

## Quick Start

Install from PyPI:

```bash
pip install vgsp
```

Transform tabular data into one dimension:

```python
from vgsp import VGSP
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=500, n_features=12, n_informative=8,
    n_redundant=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

vgsp = VGSP(n_features=7)
Z_train = vgsp.fit_transform(X_train, y_train)  # shape: (400, 1)
Z_test = vgsp.transform(X_test)                 # shape: (100, 1)

print("Selected columns:", vgsp.selected_input_indices_)
print("Projection weights:", vgsp.weights_)
```

Or install the development version from source:

```bash
git clone https://github.com/nqmn/vgsp.git
cd vgsp
python -m pip install -e ".[dev]"
```

## Features

| Category | Capability |
| --- | --- |
| Dimensionality reduction | Converts selected *k* numeric features into **one scalar** per sample |
| Supervised weighting | Bounded CVR uses between-class and within-class variance |
| Feature ranking | Deterministic top-*k* selection by descending CVR scores |
| Training-only preprocessing | Median imputation, zero-variance filtering, and min-max scaling are learned during `fit()` |
| Leakage-aware evaluation | `transform()` reuses fitted parameters; compatible with scikit-learn `Pipeline` |
| Model interoperability | Works with downstream estimators such as Decision Tree and Random Forest |
| Inspection | Exposes `cvr_scores_`, `weights_`, `selected_input_indices_`, and optional `selected_feature_names_` |
| Input | Two-class labels and numeric feature matrices; NumPy arrays or pandas DataFrames |

## Usage

### Python API

A `VGSP` instance provides the familiar scikit-learn transformer interface:

```python
from vgsp import VGSP

transformer = VGSP(n_features=7, epsilon=1e-10)
transformer.fit(X_train, y_train)
Z_train = transformer.transform(X_train)
Z_test = transformer.transform(X_test)

print(Z_train.shape)             # (number_of_training_rows, 1)
print(transformer.weights_)       # CVR-proportional weights
print(transformer.cvr_scores_)    # fitted CVR scores
```

### Combine with a Classifier

```python
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from vgsp import VGSP

model = Pipeline([
    ("vgsp", VGSP(n_features=7)),
    ("classifier", DecisionTreeClassifier(random_state=42)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Cross-Validation Without Fitting on Test Data

The research's chosen evaluation design uses a stratified **80:20 development/hold-out split** and **5-fold `StratifiedKFold`** within the 80% development data:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    model, X_train, y_train, scoring="f1_macro", cv=cv
)
print("Development CV Macro-F1:", scores.mean())

model.fit(X_train, y_train)
print("Held-out test accuracy:", model.score(X_test, y_test))
```

`Pipeline` ensures that imputation, scaling, feature ranking, CVR estimation and projection weights are fitted **inside each training fold**. The `VGSP` class itself does not choose the optimal *k* or run cross-validation automatically.

## How VGSP Works

Given two classes, class proportions `p0` and `p1`, and each feature `j`:

```text
Within-class variance:   W_j   = p0 * var(X0,j) + p1 * var(X1,j)
Between-class variance:  B_j   = p0 * p1 * (mean(X0,j) - mean(X1,j))^2
Bounded score:           CVR_j = B_j / (B_j + W_j + epsilon)
Projection weight:       w_j   = CVR_j / (sum(CVR_i for i in S_k) + epsilon)
Scalar representation:   z     = X_scaled[:, S_k] @ w
```

Here `S_k` contains the top-*k* ranked features; `epsilon` defaults to `1e-10`. The implementation uses population variance (`ddof=0`). CVR is a bounded monotonic reformulation related to the classical Fisher score; the research contribution is the **CVR-guided scalar projection**, not a claim that CVR alone is an entirely new statistical criterion.

| Method | Uses labels during fit? | Output | Primary objective |
| --- | --- | --- | --- |
| PCA (`n_components=1`) | No | 1 component | Capture variance in feature space |
| VGSP (`n_features=7`) | **Yes** | **1 scalar** | CVR-weighted class-discriminative representation |

VGSP uses a nonnegative, CVR-proportional weighted sum of selected features. It does not use signed coefficients or perform a general optimal discriminant projection.

## API Reference

| Call / attribute | Purpose |
| --- | --- |
| `VGSP(n_features=7, epsilon=1e-10)` | Configure selected feature count and numerical stabilizer |
| `fit(X, y)` | Fit preprocessing, CVR ranking and projection weights |
| `transform(X)` | Map new observations using the frozen fitted transformation |
| `fit_transform(X, y)` | Fit and return the 1D representation |
| `get_feature_names_out()` | Return `['vgsp_0']` |
| `cvr_scores_` | CVR scores of retained training features |
| `selected_input_indices_` | Selected columns relative to the original input matrix |
| `selected_feature_names_` | Selected names when fitted using string-named DataFrame columns |
| `weights_` | Normalized projection weights for selected features |

## Requirements

- Python **3.10 or later**
- NumPy **1.24 or later**
- scikit-learn **1.6 or later**, below 2.0
- pandas is optional for DataFrame inputs, and included in development dependencies

The base package is intentionally small and does not install XGBoost, LightGBM, Mininet or network traffic capture utilities.

## Project Structure

```text
vgsp/
├── pyproject.toml                 # Package metadata and build configuration
├── PYPI_README.md                 # Documentation displayed on PyPI
├── README.md                      # Research notebook / repository documentation
├── src/vgsp/
│   ├── __init__.py                # Public API
│   └── _vgsp.py                   # VGSP implementation
├── examples/basic_usage.py       # Self-contained evaluation example
├── tests/test_vgsp.py            # Automated regression tests
├── vgspv4.ipynb                   # Research notebook
└── .github/workflows/
    └── publish-pypi.yml          # PyPI publishing workflow
```

## Testing

To run the tests and local example after cloning the repository:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/basic_usage.py
```

To build the distributions locally:

```bash
python -m build
python -m twine check dist/*
```

## Scope and Limitations

- VGSP currently supports **binary supervised** feature projection, not multiclass targets.
- It expects numeric features; users must remove target labels and non-feature identifiers before fitting.
- Encoding `Protocol` as a numeric value is a research-specific assumption and is not a general categorical-encoding recommendation.
- A scalar representation is not invertible; there is no `inverse_transform()`.
- Training and classification costs can be low, but the package does **not** measure or eliminate traffic capture / upstream flow-feature-extraction overhead.
- `n_features=7` reflects the latest InSDN development configuration. It is a default, not a universally optimal value; independently choose it using development data for other tasks.
- The package contains neither datasets nor a pretrained classifier; reported research accuracy is **not** guaranteed on other datasets.

## Citation

If you use VGSP in research, cite the appropriate code version and the accompanying research article when it is available. The existing Zenodo archive documents a **research notebook snapshot**, not automatically the PyPI package release:

- [Research code archive (Zenodo DOI)](https://doi.org/10.5281/zenodo.24183611)
- [Source code and version history](https://github.com/nqmn/vgsp)

```bibtex
@software{mokti_vgsp_2026,
  author  = {Mokti, Mohd Adil},
  title   = {VGSP: Variance-Guided Scalar Projection},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/nqmn/vgsp}
}
```

## Contributing

Suggestions, bug reports and pull requests are welcome via [GitHub Issues](https://github.com/nqmn/vgsp/issues). Please include a minimal reproducing example for bugs.

## License

**Apache License 2.0** — see [LICENSE](https://github.com/nqmn/vgsp/blob/main/LICENSE).

## Project Links

- **PyPI:** https://pypi.org/project/vgsp/
- **Repository:** https://github.com/nqmn/vgsp
- **Issues:** https://github.com/nqmn/vgsp/issues
- **Research archive:** https://doi.org/10.5281/zenodo.24183611

## Changelog

### 0.1.1 — Documentation update

- Restructured the PyPI project page around quick start, feature summary, usage, testing, citation and license.
- Corrected the public installation instructions to `pip install vgsp`.
- No intentional change to the VGSP mathematical method or transformer API.
