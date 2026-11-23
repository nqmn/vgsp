# VGSP — Variance-Guided Scalar Projection

A `scikit-learn`-compatible **supervised** transformer that maps numeric feature vectors onto one scalar axis using **bounded class-variance ratio (CVR)** weights.

This implementation follows `vgspv4.ipynb` (Cell 4) and its training-only imputation, variance filtering and min–max scaling. It is **not** a pretrained DDoS detector and it does not fetch any network data.

## Install

**Before PyPI publication**, clone `https://github.com/nqmn/vgsp` and install:

```bash
python -m pip install -e ".[dev]"
```

**After successful PyPI publication:**

```bash
python -m pip install vgsp-ml
```

The distribution name is `vgsp-ml`; the Python import name is `vgsp`.

## Quick start

```python
from vgsp import VGSP
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# X: numeric features only; y: labels for two classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

model = Pipeline([
    ("vgsp", VGSP(n_features=7)),
    ("classifier", DecisionTreeClassifier(random_state=42)),
])
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.named_steps["vgsp"].selected_input_indices_)
print(model.named_steps["vgsp"].weights_)
```

Or use it directly like a `scikit-learn` transformer:

```python
vgsp = VGSP(n_features=7)
Z_train = vgsp.fit_transform(X_train, y_train)  # shape: (n_train, 1)
Z_test = vgsp.transform(X_test)                 # shape: (n_test, 1)
```

VGSP **requires labels in `fit`**; unlike PCA it is supervised. `transform` does not accept labels and never refits.

## Mathematical definition (binary labels)

For feature `j`, with class prior probabilities `p0` and `p1`, class-wise means `mu0,j` and `mu1,j`, and population variances `v0,j` and `v1,j`:

- `W_j = p0*v0,j + p1*v1,j`
- `B_j = p0*p1*(mu0,j-mu1,j)^2`
- `CVR_j = B_j/(B_j+W_j+epsilon)`
- `S_k = indices of the largest k CVR values (stable ties)`
- `w_j = CVR_j/(sum_{i in S_k} CVR_i+epsilon)`
- `z = X_scaled[:, S_k] @ w`

`epsilon` defaults to `1e-10`, and variance is the population variance (`ddof=0`).

The bounded CVR criterion is related monotonically to the Fisher score, rather than a claim of an entirely new mathematical separability measure. The research contribution is the CVR-weighted scalar projection mechanism.

## What is included inside the transformer

Each `VGSP.fit(X_train, y_train)` performs the following operations using **training data only**:

1. Converts positive/negative infinity to missing values and applies **median imputation** (`SimpleImputer(keep_empty_features=True)`).
2. Removes constant columns (`VarianceThreshold(0.0)`).
3. Learns the **MinMaxScaler**.
4. Computes **CVR**, ranks features, and freezes `k` indices and weights.

`VGSP.transform(X_valid_or_test)` applies the same fitted imputer, variance filter, scaler and projection weights without recomputing CVR. With a scikit-learn `Pipeline`, these stages are refitted only to each CV training fold.

## Evaluation protocol preserved from notebook v4

The research notebook uses a **stratified 80:20 split** and **five-fold `StratifiedKFold` on the 80% development portion**. The choice `n_features=7` reflects the latest InSDN development selection. This library **does not automatically tune k**; tuning and final evaluation stay outside the transformer and must never involve the 20% test set.

For the additional DDOSDN2025 evaluation that fixes the **same seven feature names from InSDN**, explicitly pass `X[FIXED_FEATURE_NAMES]` for that comparison. By default, `VGSP` ranks features locally within the data passed to `fit`; it does **not** claim that selected feature identities transfer between datasets automatically.

## Scope and limitations

- Binary supervised projection only, for numeric tabular features.
- `Protocol` coded numerically is a methodological assumption in the supplied research; this package does not imply ordinal meaning is generally appropriate.
- A scalar representation is not invertible; no `inverse_transform` is defined.
- A single scalar downstream input does not mean upstream data capture or flow feature-extraction cost disappears.
- The dataset is not bundled into the package; no DDoS training/evaluation results are guaranteed on other data.

## Development and testing

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/basic_usage.py
python -m build
python -m twine check dist/*
```

## Publication checklist

1. Confirm the distribution name `vgsp-ml` is available on PyPI (or choose another) and ensure it is not confused with `PyGSP`, which is an existing *graph signal processing* package.
2. Preserve the existing Apache-2.0 `LICENSE` already in the repository.
3. Confirm the source code and PyPI metadata refer to `https://github.com/nqmn/vgsp`.
4. Run tests and build the distributions.
5. Trial-publish to TestPyPI, then publish to PyPI using a verified project and preferably GitHub Actions Trusted Publishing.
6. An earlier research notebook snapshot already has a Zenodo DOI: https://doi.org/10.5281/zenodo.24183611. If citing this newly packaged release, archive its exact source version separately and verify the new version DOI. PyPI alone does not assign DOIs.

The manuscript and notebook are research references; confirm the final manuscript and published implementation are aligned before public release.
