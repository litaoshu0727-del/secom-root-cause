"""Figures for the report. Every function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BAR = "#4c6a92"
ALERT = "#c0392b"
MUTED = "#b8c0cc"
WARN = "#e0a030"


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def p_chart(weekly: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = weekly.index
    ax.plot(x, weekly["rate"] * 100, color=BAR, marker="o", label="Weekly fail rate")
    ax.step(x, weekly["ucl"] * 100, color=MUTED, where="mid", label="3-sigma limits")
    ax.step(x, weekly["lcl"] * 100, color=MUTED, where="mid")
    ax.axhline(weekly["center"].iloc[0] * 100, color=MUTED, ls="--", lw=1)
    out = weekly[weekly["out_of_control"]]
    ax.scatter(out.index, out["rate"] * 100, color=ALERT, zorder=3, s=60, label="Out of control")
    ax.set_ylabel("Fail rate (%)")
    ax.set_title("Weekly fail rate (p-chart)")
    fig.autofmt_xdate()
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    return _save(fig, path)


def volcano(values: pd.DataFrame, raw_hits: set[str], survivors: set[str], path: str | Path) -> Path:
    """Effect size against significance for every sensor, on the raw values.

    Sensors are coloured by whether they pass the screen and whether they
    still pass once the weekly median is removed.
    """
    d = values.dropna(subset=["p_value"])
    kept = d["sensor"].isin(survivors)
    lost = d["sensor"].isin(raw_hits) & ~kept
    x, yv = d["auc"], -np.log10(d["p_value"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(x[~kept & ~lost], yv[~kept & ~lost], s=12, color=MUTED, label="Not significant")
    ax.scatter(x[lost], yv[lost], s=24, color=WARN,
               label=f"Significant on raw values only ({int(lost.sum())})")
    ax.scatter(x[kept], yv[kept], s=28, color=ALERT,
               label=f"Significant within the same week ({int(kept.sum())})")
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("AUC on raw values: P(failing run reads higher than passing run)")
    ax.set_ylabel("-log10(p), raw values")
    ax.set_title("Sensor screening, fail vs pass")
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    return _save(fig, path)


def top_sensors(X: pd.DataFrame, y: pd.Series, sensors: list[str], path: str | Path) -> Path:
    """Pass vs fail distribution of the leading sensors."""
    sensors = sensors[:6]
    fig, axes = plt.subplots(1, len(sensors), figsize=(2.3 * len(sensors), 3.4), squeeze=False)
    for ax, name in zip(axes[0], sensors):
        groups = [X.loc[y == 0, name].dropna(), X.loc[y == 1, name].dropna()]
        box = ax.boxplot(groups, tick_labels=["pass", "fail"], widths=0.6, patch_artist=True,
                         showfliers=False, medianprops={"color": "black"})
        for patch, color in zip(box["boxes"], (MUTED, ALERT)):
            patch.set_facecolor(color)
        ax.set_title(name, fontsize=10)
    fig.suptitle("Leading sensors, weekly median removed: pass vs fail (outliers hidden)", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def stability(freq: pd.Series, path: str | Path, top: int = 20) -> Path:
    d = freq.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(d.index, d.to_numpy() * 100, color=BAR)
    ax.set_xlabel("Selected in % of random half-samples")
    ax.set_title(f"Stability selection, top {top} sensors")
    return _save(fig, path)
