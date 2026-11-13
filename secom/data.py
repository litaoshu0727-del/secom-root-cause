"""Load the UCI SECOM files and do the basic column clean-up."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

TIME_FORMAT = "%d/%m/%Y %H:%M:%S"


def load_secom(folder: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, t) sorted by time.

    X: one column per sensor, named s000..s589.
    y: 1 = fail, 0 = pass (the raw file uses 1 = fail, -1 = pass).
    t: test timestamp of each run.
    """
    folder = Path(folder)
    X = pd.read_csv(folder / "secom.data", sep=r"\s+", header=None, na_values="NaN")
    X.columns = [f"s{i:03d}" for i in range(X.shape[1])]
    labels = pd.read_csv(folder / "secom_labels.data", sep=" ", header=None,
                         names=["label", "time"], quotechar='"')
    if len(labels) != len(X):
        raise ValueError(f"{len(X)} sensor rows but {len(labels)} label rows")
    t = pd.to_datetime(labels["time"], format=TIME_FORMAT)
    order = np.argsort(t.to_numpy(), kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    y = (labels["label"] == 1).astype(int).iloc[order].reset_index(drop=True).rename("fail")
    return X, y, t.iloc[order].reset_index(drop=True).rename("time")


def clean(X: pd.DataFrame, max_missing: float = 0.5) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop sensors that cannot carry information: constant, or mostly missing."""
    constant = X.nunique(dropna=True) <= 1
    too_missing = (X.isna().mean() > max_missing) & ~constant
    kept = X.loc[:, ~constant & ~too_missing]
    summary = {
        "n_runs": len(X),
        "n_sensors": X.shape[1],
        "dropped_constant": int(constant.sum()),
        "dropped_missing": int(too_missing.sum()),
        "kept": kept.shape[1],
    }
    return kept, summary


def correlated_groups(X: pd.DataFrame, threshold: float = 0.95) -> pd.Series:
    """Group sensors that move together (|Spearman| >= threshold, linked transitively).

    Near-duplicate sensors would otherwise fill the shortlist with several
    copies of one signal. Returns sensor -> name of the first sensor in its group.
    """
    corr = X.corr(method="spearman").abs().fillna(0).to_numpy()
    _, labels = connected_components(csr_matrix(corr >= threshold), directed=False)
    first = pd.Series(X.columns, index=X.columns).groupby(labels).transform("first")
    return first.rename("group")
