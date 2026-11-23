"""The variance-guided scalar projection (VGSP) transformer.

The CVR calculations and train-only preprocessing follow the research
implementation supplied in notebook vgspv4.ipynb (Cell 4) and associated
train-only preprocessing cells. The original model is binary supervised.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import (
    check_consistent_length,
    check_is_fitted,
    check_array,
)


class VGSP(TransformerMixin, BaseEstimator):
    """Project numeric features onto one CVR-weighted scalar axis.

    VGSP is a *supervised* dimensionality-reduction transformer. Its fit()
    requires binary labels. Preprocessing (median imputation, removal of
    constant features, and min-max scaling) is learned only from fit(X, y)
    and reused unchanged during transform(X).

    Parameters
    ----------
    n_features : int, default=7
        Number of top-ranked features to use. The research's latest
        validated InSDN configuration chose seven features; this value
        can be changed for other applications.
    epsilon : float, default=1e-10
        Strictly positive numerical stabilizer for CVR and weight sums.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Class labels ordered by np.unique (class mapping is internal).
    cvr_scores_ : ndarray of shape (n_informative_features,)
        CVR scores on the training data, after variance filtering.
    selected_indices_ : ndarray of shape (n_features,)
        Selected column indices after variance filtering.
    selected_input_indices_ : ndarray of shape (n_features,)
        Selected indices referring to the original input columns.
    weights_ : ndarray of shape (n_features,)
        CVR-proportional projection weights.
    selected_feature_names_ : ndarray of shape (n_features,)
        Names when fitted on a pandas DataFrame with string column names.

    Notes
    -----
    The per-feature CVR and scalar projection use the research equations:
        W = p0*var(X0) + p1*var(X1)
        B = p0*p1*(mean(X0)-mean(X1))**2
        CVR = B/(B+W+epsilon)
        w = CVR_selected/(sum(CVR_selected)+epsilon)
        z = X_scaled_selected @ w
    Population variance is used (numpy default, ddof=0).
    Only binary classification is supported by this research version.

    Example
    -------
    >>> from vgsp import VGSP
    >>> model = VGSP(n_features=2)
    >>> X = [[1., 4., 0], [2., 5., 2], [6., 7., 3], [7., 8., 5]]
    >>> z = model.fit_transform(X, [0, 0, 1, 1])
    >>> z.shape
    (4, 1)
    """

    def __init__(self, n_features: int = 7, epsilon: float = 1e-10):
        self.n_features = n_features
        self.epsilon = epsilon

    def _validate_input(self, X, *, reset: bool):
        # Preserve DataFrame names after validate_data: converting to an ndarray
        # first makes scikit-learn clear feature_names_in_ during a fit reset.
        pending_names = None
        if hasattr(X, "columns"):
            col_names = list(X.columns)
            if len(set(col_names)) != len(col_names):
                raise ValueError("Input DataFrame has duplicate column names.")
            if all(isinstance(c, str) for c in col_names):
                pending_names = np.asarray(col_names, dtype=object)
                if not reset and hasattr(self, "feature_names_in_") and not np.array_equal(
                    self.feature_names_in_, pending_names
                ):
                    raise ValueError(
                        "Input DataFrame columns do not match the training columns "
                        "in their original order."
                    )
            elif not reset and hasattr(self, "feature_names_in_"):
                raise ValueError("Expected DataFrame with original string columns.")

        try:
            values = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "VGSP requires numeric features. Exclude identifiers and labels first."
            ) from exc

        if values.ndim != 2:
            raise ValueError("VGSP expects a two-dimensional feature matrix.")
        # Notebook Cell 2 replaces +/-infinity by NaN before train-only
        # imputation. Replicate that conversion within the transformer.
        values = np.where(np.isfinite(values), values, np.nan)
        checked = check_array(
            values, dtype=np.float64, ensure_all_finite="allow-nan",
            ensure_min_samples=1,
        )
        if reset:
            self.n_features_in_ = checked.shape[1]
        elif checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} input features, got {checked.shape[1]}."
            )
        if reset and pending_names is not None:
            self.feature_names_in_ = pending_names
        return checked

    def fit(self, X, y):
        """Fit train-only preprocessing, class CVR scores and 1-D weights."""
        if isinstance(self.n_features, (bool, np.bool_)) or not isinstance(
            self.n_features, (int, np.integer)
        ) or self.n_features < 1:
            raise ValueError("n_features must be a positive integer.")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number.")
        if y is None:
            raise ValueError("VGSP is supervised: fit(X, y) requires binary labels.")

        A = self._validate_input(X, reset=True)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional binary label vector.")
        check_consistent_length(A, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("VGSP requires exactly two classes in y.")

        self.imputer_ = SimpleImputer(
            strategy="median", keep_empty_features=True
        )
        A = self.imputer_.fit_transform(A)
        self.variance_filter_ = VarianceThreshold(threshold=0.0)
        try:
            A = self.variance_filter_.fit_transform(A)
        except ValueError as exc:
            raise ValueError(
                "No non-constant numeric features remain after imputation."
            ) from exc
        if self.n_features > A.shape[1]:
            raise ValueError(
                f"n_features={self.n_features} exceeds the {A.shape[1]} "
                "non-constant training features."
            )
        self.scaler_ = MinMaxScaler()
        A = self.scaler_.fit_transform(A)

        X0 = A[y == self.classes_[0]]
        X1 = A[y == self.classes_[1]]
        p0 = X0.shape[0] / A.shape[0]
        p1 = X1.shape[0] / A.shape[0]
        within = p0 * X0.var(axis=0) + p1 * X1.var(axis=0)
        between = p0 * p1 * (X0.mean(axis=0) - X1.mean(axis=0)) ** 2

        self.class_priors_ = np.array([p0, p1])
        self.within_variance_ = within
        self.between_variance_ = between
        self.cvr_scores_ = between / (between + within + self.epsilon)
        self.selected_indices_ = np.argsort(
            -self.cvr_scores_, kind="stable"
        )[: self.n_features]
        selected_scores = self.cvr_scores_[self.selected_indices_]
        self.weights_ = selected_scores / (selected_scores.sum() + self.epsilon)

        retained_raw = self.variance_filter_.get_support(indices=True)
        self.selected_input_indices_ = retained_raw[self.selected_indices_]
        if hasattr(self, "feature_names_in_"):
            self.selected_feature_names_ = self.feature_names_in_[
                self.selected_input_indices_
            ]
        return self

    def transform(self, X):
        """Apply the *frozen* train-fitted projection to new examples."""
        check_is_fitted(self, ["imputer_", "variance_filter_", "scaler_", "weights_"])
        A = self._validate_input(X, reset=False)
        A = self.imputer_.transform(A)
        A = self.variance_filter_.transform(A)
        A = self.scaler_.transform(A)
        z = A[:, self.selected_indices_] @ self.weights_
        return z.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        """Return the one-dimensional output feature's name."""
        check_is_fitted(self, "weights_")
        if input_features is not None:
            provided = list(input_features)
            if len(provided) != self.n_features_in_:
                raise ValueError("Incorrect number of input feature names.")
            if hasattr(self, "feature_names_in_") and not np.array_equal(
                self.feature_names_in_, provided
            ):
                raise ValueError("input_features differ from fitted feature names.")
        return np.asarray(["vgsp_0"], dtype=object)

    def __sklearn_tags__(self):
        """Advertise required labels and support for missing input values."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = True
        return tags
