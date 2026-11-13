"""Generate a synthetic dataset in the exact SECOM file format, with known planted causes.

Used by the tests (the answer is known, so the screening can be checked) and
to run the pipeline before the real data is available. Results on it say
nothing about the real SECOM data.

    python -m secom.synthetic --out data/synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TIME_FORMAT


def make_secom(n: int = 1567, p: int = 590, seed: int = 0):
    """Return (X, y, t, truth). `truth` lists which columns were planted as what."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2008-07-19")
    weeks = 10
    t = start + pd.to_timedelta(np.sort(rng.uniform(0, weeks * 7 * 24 * 3600, n)), unit="s")
    week = ((t - start).days // 7).to_numpy()

    # Fails: 5% normally, 15% during a two-week excursion.
    excursion_weeks = [6, 7]
    y = (rng.random(n) < np.where(np.isin(week, excursion_weeks), 0.15, 0.05)).astype(int)

    cols = rng.permutation(p)
    signal, copies_of = cols[:8], cols[8:14]          # 6 near-copies of the first 3 signals
    drift, constant = cols[14:34], cols[34:134]
    mostly_missing, informative_missing = cols[134:154], cols[154:156]

    scale, loc = rng.uniform(0.5, 50, p), rng.uniform(-100, 100, p)
    Z = rng.standard_normal((n, p))
    Z[np.ix_(y == 1, signal)] += rng.uniform(0.8, 1.2, len(signal))   # true causes shift on fails
    for i, col in enumerate(copies_of):
        Z[:, col] = Z[:, signal[i // 2]] + 0.1 * rng.standard_normal(n)
    Z[:, drift] += np.linspace(-1, 1, n)[:, None]                      # slow drift, unrelated to fails
    X = Z * scale + loc
    X[:, constant] = loc[constant]

    X[rng.random((n, p)) < 0.03] = np.nan
    X[:, mostly_missing] = np.where(rng.random((n, len(mostly_missing))) < 0.75, np.nan,
                                    X[:, mostly_missing])
    for col in informative_missing:                                    # sensor drops out on fails
        X[rng.random(n) < np.where(y == 1, 0.6, 0.1), col] = np.nan

    name = lambda idx: [f"s{i:03d}" for i in idx]
    truth = {
        "signal": name(signal), "copies": name(copies_of), "drift": name(drift),
        "constant": name(constant), "mostly_missing": name(mostly_missing),
        "informative_missing": name(informative_missing), "excursion_weeks": excursion_weeks,
    }
    return (pd.DataFrame(X, columns=name(range(p))), pd.Series(y, name="fail"),
            pd.Series(t, name="time"), truth)


def write_secom(folder: str | Path, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    X.to_csv(folder / "secom.data", sep=" ", header=False, index=False, na_rep="NaN")
    with open(folder / "secom_labels.data", "w") as f:
        for label, stamp in zip(y, t):
            f.write(f'{1 if label else -1} "{stamp.strftime(TIME_FORMAT)}"\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    X, y, t, _ = make_secom(seed=args.seed)
    write_secom(args.out, X, y, t)
    print(f"wrote {len(X)} runs x {X.shape[1]} sensors to {args.out}  ({int(y.sum())} fails)")


if __name__ == "__main__":
    main()
