import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from vgsp import VGSP


def data():
    # Deliberately different scales, a missing value, and a constant feature.
    X = pd.DataFrame({
        "flow": [0., 1., 2., 3., 8., 9., 10., 11.],
        "packets": [3., np.nan, 5., 7., 2., 4., 5., 9.],
        "constant": [5.] * 8,
        "rate": [0., 1., 0., 1., 2., 2., 3., 3.],
    })
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def test_scalar_output_and_normalized_score():
    X, y = data()
    m = VGSP(n_features=2).fit(X, y)
    Z = m.transform(X)
    assert Z.shape == (8, 1)
    assert np.all((0 <= m.cvr_scores_) & (m.cvr_scores_ < 1))
    assert m.weights_.sum() <= 1
    assert_array_equal(m.get_feature_names_out(), ["vgsp_0"])
    assert len(m.selected_input_indices_) == 2
    assert "constant" not in m.selected_feature_names_


def test_formula_and_projection_match_notebook_cell_4():
    X, y = data()
    m = VGSP(n_features=2).fit(X, y)
    # Independent notebook-equivalent formula on the fitted preprocessing output.
    A = m.scaler_.transform(m.variance_filter_.transform(
        m.imputer_.transform(X.to_numpy())
    ))
    A0, A1 = A[y == 0], A[y == 1]
    p0, p1 = len(A0) / len(A), len(A1) / len(A)
    W = p0 * A0.var(axis=0) + p1 * A1.var(axis=0)
    B = p0 * p1 * (A0.mean(axis=0) - A1.mean(axis=0)) ** 2
    cvr = B / (B + W + 1e-10)
    idx = np.argsort(-cvr, kind="stable")[:2]
    weights = cvr[idx] / (cvr[idx].sum() + 1e-10)
    assert_allclose(m.cvr_scores_, cvr, rtol=0, atol=1e-15)
    assert_array_equal(m.selected_indices_, idx)
    assert_allclose(m.weights_, weights, rtol=0, atol=1e-15)
    assert_allclose(m.transform(X).ravel(), A[:, idx] @ weights, atol=1e-15)


def test_no_refit_during_transform():
    X, y = data()
    m = VGSP(n_features=2).fit(X, y)
    old_weights = m.weights_.copy()
    old_scores = m.cvr_scores_.copy()
    extreme = X.iloc[:2].copy()
    extreme.loc[:, "flow"] = 1e8
    m.transform(extreme)
    assert_array_equal(m.weights_, old_weights)
    assert_array_equal(m.cvr_scores_, old_scores)


def test_column_order_is_frozen():
    X, y = data()
    m = VGSP(n_features=2).fit(X, y)
    with pytest.raises(ValueError, match="columns do not match"):
        m.transform(X[list(reversed(X.columns))])


def test_fit_requires_binary_labels():
    X, y = data()
    with pytest.raises(ValueError, match="exactly two classes"):
        VGSP(n_features=2).fit(X, np.zeros_like(y))
    with pytest.raises(ValueError, match="requires binary labels"):
        VGSP(n_features=2).fit(X, None)


def test_too_many_features_rejected():
    X, y = data()
    with pytest.raises(ValueError, match="exceeds"):
        VGSP(n_features=4).fit(X, y)


def test_pipeline_cross_validation_and_dataframe_output():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(150, 9))
    y = np.repeat([0, 1], 75)
    X[y == 1, 0] += 2.0
    model = Pipeline([
        ("vgsp", VGSP(n_features=7)),
        ("dt", DecisionTreeClassifier(random_state=42)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    assert scores.shape == (5,)
    assert np.isfinite(scores).all()
    out = VGSP(n_features=7).set_output(transform="pandas").fit_transform(X, y)
    assert list(out.columns) == ["vgsp_0"]
    assert out.shape == (150, 1)


def test_infinity_is_imputed_using_training_data():
    X, y = data()
    m = VGSP(n_features=2).fit(X, y)
    invalid = X.iloc[:2].copy()
    invalid.loc[:, "flow"] = np.inf
    assert np.isfinite(m.transform(invalid)).all()
