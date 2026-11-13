"""Time structure: is the fail rate stable, and which sensors drift."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def weekly_fail_rate(y: pd.Series, t: pd.Series) -> pd.DataFrame:
    """p-chart of the weekly fail rate with 3-sigma control limits.

    Limits depend on each week's run count, so a quiet week gets wider limits.
    """
    week = t.dt.to_period("W").dt.start_time
    out = y.groupby(week.to_numpy()).agg(n="size", fails="sum")
    out.index.name = "week"
    out["rate"] = out["fails"] / out["n"]
    center = y.mean()
    sigma = np.sqrt(center * (1 - center) / out["n"])
    out["center"] = center
    out["ucl"] = center + 3 * sigma
    out["lcl"] = (center - 3 * sigma).clip(lower=0)
    out["out_of_control"] = (out["rate"] > out["ucl"]) | (out["rate"] < out["lcl"])
    return out


def half_split_fail_rates(y: pd.Series) -> dict[str, float]:
    """Fail rate in the first vs second half of the runs (time-sorted), with a Fisher exact p-value."""
    half = len(y) // 2
    first, second = y.iloc[:half], y.iloc[half:]
    table = [[int(first.sum()), int(len(first) - first.sum())],
             [int(second.sum()), int(len(second) - second.sum())]]
    return {"first_half": float(first.mean()), "second_half": float(second.mean()),
            "p_value": float(fisher_exact(table)[1])}


def center_by_week(X: pd.DataFrame, t: pd.Series) -> pd.DataFrame:
    """Subtract each sensor's weekly median, leaving only within-week variation.

    Screening the result asks "among runs made in the same week, do failing
    runs read differently?", which a slow drift cannot fake. It is
    conservative: a real cause that only changes between weeks is removed too.
    """
    week = t.dt.to_period("W").astype(str).to_numpy()
    return X - X.groupby(week).transform("median")


def sensor_drift(X: pd.DataFrame) -> pd.Series:
    """Spearman correlation of each sensor with run order (rows must be time-sorted).

    A sensor that drifts over time will correlate with anything else that
    changes over time, including the fail rate, without causing it.
    """
    order = pd.Series(np.arange(len(X)), index=X.index)
    return X.corrwith(order, method="spearman").rename("drift_rho")
