# VGSP — Variance-Guided Scalar Projection

[![PyPI](https://img.shields.io/pypi/v/vgsp.svg)](https://pypi.org/project/vgsp/)
[![Python](https://img.shields.io/pypi/pyversions/vgsp.svg)](https://pypi.org/project/vgsp/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24183611.svg)](https://doi.org/10.5281/zenodo.24183611)

**A lightweight, scikit-learn-compatible supervised feature transformer for one-dimensional representation.**

VGSP (Variance-Guided Scalar Projection) uses a **bounded Class-Variance Ratio (CVR)** to rank input features, compute normalized nonnegative projection weights, and map multidimensional numerical observations onto **one scalar feature**. It originated in research on lightweight DDoS detection for Software-Defined Networks (SDN), but can be applied to suitable binary-labelled tabular datasets.

VGSP is **supervised** (unlike PCA): training requires `fit(X, y)`. It is a transformer, **not** a pretrained DDoS detector.

## Quick Start

Install the published package:

```bash
pip install vgsp
```

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
Z_train = vgsp.fit_transform(X_train, y_train)  # (400, 1)
Z_test = vgsp.transform(X_test)                 # (100, 1)

print(vgsp.selected_input_indices_)
print(vgsp.weights_)
```

## Features

| Capability | Description |
| --- | --- |
| Supervised 1D projection | Maps selected top-*k* numeric features to a single scalar |
| Bounded CVR | Uses between-class and within-class variance to rank features |
| Train-only preprocessing | Median imputation, zero-variance filtering, and min-max scaling learned in `fit()` |
| Leakage-aware usage | `transform()` reuses frozen training estimates; integrates with `Pipeline` |
| scikit-learn interface | `fit`, `transform`, `fit_transform`, and `get_feature_names_out` |
| Model-agnostic | Suitable for downstream classifiers such as DT, RF, or logistic regression |

### Use VGSP in a scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from vgsp import VGSP

model = Pipeline([
    ("vgsp", VGSP(n_features=7)),
    ("classifier", DecisionTreeClassifier(random_state=42)),
])
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

The research evaluation protocol uses a stratified 80:20 development/hold-out split and 5-fold `StratifiedKFold` on the development portion. Use a `Pipeline` in each fold so all preprocessing, CVR estimates, rankings and weights are fitted on **training data only**. The transformer does not automatically choose the optimal *k*.

## How VGSP Works

For each feature *j* in a binary-labelled training dataset, let *p₀*, *p₁* be class proportions, *μ₀ⱼ*, *μ₁ⱼ* class means and *v₀ⱼ*, *v₁ⱼ* population variances:

\[
W_j = p_0 v_{0,j} + p_1 v_{1,j}, \qquad
B_j = p_0 p_1 (\mu_{0,j}-\mu_{1,j})^2
\]

\[
\mathrm{CVR}_j = \frac{B_j}{B_j+W_j+\varepsilon},\qquad
w_j = \frac{\mathrm{CVR}_j}{\sum_{i \in S_k} \mathrm{CVR}_i+\varepsilon}
\]

\[
z = X_{\mathrm{scaled},S_k}w
\]

Here, *Sₖ* denotes the top-*k* features ranked by CVR; `epsilon=1e-10` by default. CVR is a bounded monotonic reformulation related to the classical Fisher score. **The research contribution is the CVR-guided scalar projection workflow**, not a claim that the CVR ratio alone is a new standalone mathematical criterion. This is a nonnegative weighted projection, not an optimal signed discriminant direction.

## API at a Glance

| API | Description |
| --- | --- |
| `VGSP(n_features=7, epsilon=1e-10)` | Initialize transformer |
| `.fit(X, y)` | Learn imputer, variance filter, scaler, CVR, ranking and weights |
| `.transform(X)` | Return frozen 1D projection for new data |
| `.fit_transform(X, y)` | Fit and transform training data |
| `.cvr_scores_` | CVR scores of retained features |
| `.selected_input_indices_` | Selected indices in original input feature space |
| `.selected_feature_names_` | Selected feature names for suitably named DataFrames |
| `.weights_` | Normalized projection weights |

See [the PyPI package documentation](https://pypi.org/project/vgsp/) for complete usage and API examples.

## Research Notebook and Reproducibility

The repository also includes [`vgspv4.ipynb`](vgspv4.ipynb), the research notebook accompanying the manuscript *A Variance-Guided Scalar Traffic Representation for Lightweight DDoS Detection in Software-Defined Networks*.

- **Archived research software DOI:** [10.5281/zenodo.24183611](https://doi.org/10.5281/zenodo.24183611)
- **Research notebook version:** v4 (last-update note: 13 November 2026)
- **Notebook execution environment:** Python 3.12 with additional scientific, visualization and classifier dependencies (including pandas, SciPy, XGBoost and LightGBM)
- **Library installation requirement:** Python 3.10+, NumPy 1.24+, scikit-learn 1.6 to below 2.0

### Dataset inputs

The notebook expects **prepared binary CSV files** (not raw dataset downloads). The inputs are not distributed with the Python package.

| Dataset | Expected input file | Rows in notebook | Numeric features used |
| --- | --- | ---: | ---: |
| InSDN | `insdn_ddos_binary_0n1d.csv` | 190,366 | 77 |
| DDOSDN2025 | `ddosdn2025_binary_0n1d.csv` | 200,378 | 77 |

Adjust `INSDN_PATH` and `DDOSDN_PATH` in the notebook to match local file locations.

**Evaluation boundaries:** The notebook uses a stratified 80:20 hold-out split and five-fold `StratifiedKFold`, **not group-aware cross-validation**. Fitted preprocessing and CVR/weights are estimated on training partitions only. InSDN development experiments select seven features: `Protocol`, `Down/Up Ratio`, `Init Bwd Win Byts`, `Pkt Len Min`, `ACK Flag Cnt`, `Bwd Pkt Len Min`, and `Bwd Pkts/s`. In the DDOSDN2025 evaluation, the **feature names** are carried over but its preprocessing, projection weights and classifiers are independently fitted to the training portion of that dataset. **This does not constitute train-on-InSDN/test-on-DDOSDN2025 transfer evaluation.**

Dataset references:

1. Elsayed, M. S., Le-Khac, N.-A., and Jurcut, A. D. (2020). *InSDN: A Novel SDN Intrusion Dataset*. *IEEE Access*, 8, 165263–165284. https://doi.org/10.1109/ACCESS.2020.3022633
2. Mokti, M. A. (2025). *DDOSDN2025*. Mendeley Data. https://doi.org/10.17632/9c5yjbk354.1

### Limitations

- The packaged transformer supports numeric, **binary-labelled** inputs; not multiclass labels.
- The `Protocol` field's numerical coding is a research assumption, not a general recommendation to treat nominal categories as continuous.
- Stratified CV may retain dependencies among records sharing attack sessions, capture conditions, hosts, or identical traffic vectors.
- The package does not include flow capture/extraction, attack mitigation, a pretrained model or a guarantee of detection accuracy.
- Classification timing for a precomputed representation is **not** equivalent to full packet-capture-to-controller latency.
- The repository DOI above archives a research notebook snapshot. A later PyPI package release should be archived separately for a version-specific software DOI if cited independently.

## Development

```bash
git clone https://github.com/nqmn/vgsp.git
cd vgsp
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/basic_usage.py
python -m build
python -m twine check dist/*
```

## Project Structure

```text
vgsp/
├── pyproject.toml
├── README.md                  # GitHub overview and research provenance
├── PYPI_README.md             # Description published on PyPI
├── LICENSE
├── src/vgsp/
│   ├── __init__.py
│   └── _vgsp.py
├── tests/test_vgsp.py
├── examples/basic_usage.py
├── vgspv4.ipynb
└── .github/workflows/publish-pypi.yml
```

## Citation

For research use, cite the relevant version of VGSP together with the associated manuscript where applicable. The current Zenodo DOI points to the **notebook research archive**, not necessarily the identical source tree of the PyPI release.

**Existing research archive:** https://doi.org/10.5281/zenodo.24183611

**Associated manuscript:** Mokti, M. A., Abu Bakar, K. A., and Zainol Ariffin, K. A. *A Variance-Guided Scalar Traffic Representation for Lightweight DDoS Detection in Software-Defined Networks* (manuscript under revision; add journal DOI once published).

Please cite the original InSDN and DDOSDN2025 sources when using these datasets.

## License

This repository is distributed under the **Apache License 2.0**; see [LICENSE](LICENSE). Dataset licensing is independent of the package license.

## Project Links

- [PyPI: vgsp](https://pypi.org/project/vgsp/)
- [GitHub source and issues](https://github.com/nqmn/vgsp)
- [Research notebook archive (Zenodo)](https://doi.org/10.5281/zenodo.24183611)
