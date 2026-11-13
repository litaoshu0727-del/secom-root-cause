"""Univariate screening: which sensors differ between passing and failing runs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact, mannwhitneyu

MIN_PER_CLASS = 5


def _add_q_values(table: pd.DataFrame) -> pd.DataFrame:
    """Benjamini-Hochberg q-values: with hundreds of sensors tested, raw p-values
    would flag dozens by chance alone."""
    table["q_value"] = np.nan
    tested = table["p_value"].notna()
    if tested.any():
        table.loc[tested, "q_value"] = false_discovery_control(table.loc[tested, "p_value"], method="bh")
    return table.sort_values("p_value").reset_index(drop=True)


def screen_values(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Mann-Whitney U test per sensor, fail vs pass.

    `auc` is the probability that a failing run reads higher than a passing
    run (0.5 = no difference). Rank-based, so outliers and units do not matter.
    """
    fail = y.to_numpy() == 1
    rows = []
    for col in X.columns:
        v = X[col].to_numpy(dtype=float)
        seen = ~np.isnan(v)
        a, b = v[seen & fail], v[seen & ~fail]
        row = {"sensor": col, "n_fail": len(a), "n_pass": len(b), "auc": np.nan, "p_value": np.nan,
               "median_fail": np.nan, "median_pass": np.nan}
        if len(a) >= MIN_PER_CLASS and len(b) >= MIN_PER_CLASS:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            row.update(auc=u / (len(a) * len(b)), p_value=p,
                       median_fail=np.median(a), median_pass=np.median(b))
        rows.append(row)
    table = pd.DataFrame(rows)
    table["effect"] = (table["auc"] - 0.5).abs()
    return _add_q_values(table)


def screen_missingness(X: pd.DataFrame, y: pd.Series, min_missing: int = 10) -> pd.DataFrame:
    """Fisher exact test per sensor: is a missing reading more common on failing runs?

    A sensor that drops out when something goes wrong is a signal that
    imputation would erase, so it is tested separately on the raw data.
    """
    fail = y.to_numpy() == 1
    rows = []
    for col in X.columns:
        missing = X[col].isna().to_numpy()
        if not (min_missing <= missing.sum() <= len(missing) - min_missing):
            continue
        table = [[int((missing & fail).sum()), int((~missing & fail).sum())],
                 [int((missing & ~fail).sum()), int((~missing & ~fail).sum())]]
        odds, p = fisher_exact(table)
        rows.append({"sensor": col, "missing_rate_fail": missing[fail].mean(),
                     "missing_rate_pass": missing[~fail].mean(), "odds_ratio": odds, "p_value": p})
    columns = ["sensor", "missing_rate_fail", "missing_rate_pass", "odds_ratio", "p_value"]
    return _add_q_values(pd.DataFrame(rows, columns=columns))
