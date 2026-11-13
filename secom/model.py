"""Multivariate models: how predictable fails are, and which sensors a model keeps picking."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

L1_C = 0.02  # strength of the L1 penalty (smaller = fewer sensors kept)


def _l1_logistic(C: float = L1_C) -> LogisticRegression:
    return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, class_weight="balanced")


def make_models(seed: int = 0) -> dict[str, Pipeline]:
    """Every step, including imputation and sensor selection, sits inside the
    pipeline so it is refit on each training fold. Selecting sensors on the
    full data first would leak the test labels into the model."""
    prep = lambda: (SimpleImputer(strategy="median"), StandardScaler())
    return {
        "logistic_l1": make_pipeline(*prep(), _l1_logistic()),
        # VarianceThreshold: a sensor can be constant inside one training fold
        "top40_logistic": make_pipeline(*prep(), VarianceThreshold(), SelectKBest(f_classif, k=40),
                                        LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                   class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
    }


def cross_validated_scores(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> pd.DataFrame:
    """ROC-AUC and PR-AUC over 5-fold x 3 stratified CV (mean and std across the 15 folds)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    rows = []
    for name, model in make_models(seed).items():
        res = cross_validate(model, X, y, cv=cv, scoring=("roc_auc", "average_precision"))
        rows.append({"model": name,
                     "roc_auc": res["test_roc_auc"].mean(), "roc_auc_std": res["test_roc_auc"].std(),
                     "pr_auc": res["test_average_precision"].mean(),
                     "pr_auc_std": res["test_average_precision"].std()})
    return pd.DataFrame(rows).set_index("model")


def time_only_score(y: pd.Series, seed: int = 0) -> tuple[float, float]:
    """ROC-AUC, under the same random CV, of a model that sees only the run order.

    This is the score available from knowing *when* a run happened and nothing
    about its sensors. Sensor models have to beat it to show they carry
    information beyond the time trend in the fail rate.
    """
    order = pd.DataFrame({"order": np.arange(len(y))})
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=25, n_jobs=-1, random_state=seed)
    scores = cross_validate(clf, order, y, cv=cv, scoring="roc_auc")["test_score"]
    return float(scores.mean()), float(scores.std())


def time_holdout_scores(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.7, seed: int = 0) -> pd.DataFrame:
    """Train on the earliest runs, test on the latest (rows must be time-sorted).

    This is how the model would be used in a fab. If it scores well below
    random CV, the model was leaning on patterns that do not carry forward.
    """
    cut = int(len(X) * train_frac)
    X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    rows = []
    for name, model in make_models(seed).items():
        prob = model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        rows.append({"model": name, "roc_auc": roc_auc_score(y_te, prob),
                     "pr_auc": average_precision_score(y_te, prob)})
    out = pd.DataFrame(rows).set_index("model")
    out.attrs.update(n_train=len(y_tr), n_test=len(y_te), fails_train=int(y_tr.sum()),
                     fails_test=int(y_te.sum()), test_fail_rate=float(y_te.mean()))
    return out


def stability_selection(X: pd.DataFrame, y: pd.Series, n_rounds: int = 100, C: float = L1_C,
                        seed: int = 0) -> tuple[pd.Series, float]:
    """How often each sensor survives an L1 model fit on a random half of the runs.

    With 104 fails and hundreds of sensors, a single fit picks partly by
    luck. A sensor selected in most random halves is a more trustworthy lead.
    Returns (selection frequency per sensor, mean number of sensors kept per round).
    """
    splitter = StratifiedShuffleSplit(n_splits=n_rounds, train_size=0.5, random_state=seed)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), _l1_logistic(C))
    counts = np.zeros(X.shape[1])
    for half, _ in splitter.split(X, y):
        counts += model.fit(X.iloc[half], y.iloc[half])[-1].coef_.ravel() != 0
    freq = pd.Series(counts / n_rounds, index=X.columns, name="stability")
    return freq, float(counts.sum() / n_rounds)
