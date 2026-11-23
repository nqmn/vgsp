"""Basic, self-contained demonstration; no private datasets required."""
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from vgsp import VGSP

X, y = make_classification(
    n_samples=1500, n_features=12, n_informative=8, n_redundant=2,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
model = Pipeline([
    ("vgsp", VGSP(n_features=7)),
    ("dt", DecisionTreeClassifier(random_state=42)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    model, X_train, y_train, cv=cv, scoring="f1_macro"
)
model.fit(X_train, y_train)
print("5-fold development Macro-F1:", cv_scores.mean(), "+/-", cv_scores.std())
print("20% independent test accuracy:", model.score(X_test, y_test))
print("Selected raw feature indices:", model.named_steps["vgsp"].selected_input_indices_)
print("CVR weights:", model.named_steps["vgsp"].weights_)
print("Projected test shape:", model.named_steps["vgsp"].transform(X_test).shape)
