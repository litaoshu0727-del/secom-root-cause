"""Run the whole analysis and write tables and figures.

    python -m secom.report --data data/secom --out reports/secom
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import drift, model, plots, screening
from .data import clean, correlated_groups, load_secom

DRIFT_FLAG = 0.3  # |Spearman with run order| above which a sensor is marked as drifting


def build_shortlist(values: pd.DataFrame, within_week: pd.DataFrame, stability: pd.Series,
                    drift_rho: pd.Series, groups: pd.Series, q_threshold: float) -> pd.DataFrame:
    """Every sensor that passes the FDR screen on raw values or on within-week values.

    `survives` marks the ones still significant after the weekly median is
    removed: those cannot be explained by a time trend shared with the fail
    rate. The other columns are the remaining checks a reader should make:
    duplicate sensors (`group`), model stability, and drift.
    """
    raw = values.set_index("sensor")
    adj = within_week.set_index("sensor")
    out = pd.DataFrame({
        "auc": raw["auc"], "q_value": raw["q_value"],
        "auc_within_week": adj["auc"], "q_within_week": adj["q_value"], "p_within_week": adj["p_value"],
    })
    out = out[(out["q_value"] < q_threshold) | (out["q_within_week"] < q_threshold)]
    out["survives"] = out["q_within_week"] < q_threshold
    out["stability"] = stability.reindex(out.index)
    out["drift_rho"] = drift_rho.reindex(out.index)
    out["drifting"] = out["drift_rho"].abs() > DRIFT_FLAG
    out["group"] = groups.reindex(out.index)
    out = out.sort_values("p_within_week").drop(columns="p_within_week")
    out["best_in_group"] = ~out.duplicated("group")
    return out.rename_axis("sensor").reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="folder with secom.data and secom_labels.data")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--q", type=float, default=0.10, help="false discovery rate for the shortlist")
    ap.add_argument("--rounds", type=int, default=100, help="stability selection rounds")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X_raw, y, t = load_secom(args.data)
    X, summary = clean(X_raw)
    print(f"{summary['n_runs']} runs, {int(y.sum())} fails ({y.mean():.1%}), "
          f"{t.min():%Y-%m-%d} to {t.max():%Y-%m-%d}")
    print(f"sensors: {summary['n_sensors']} total, {summary['dropped_constant']} constant, "
          f"{summary['dropped_missing']} over 50% missing, {summary['kept']} kept")

    weekly = drift.weekly_fail_rate(y, t)
    halves = drift.half_split_fail_rates(y)
    X_week = drift.center_by_week(X, t)
    values = screening.screen_values(X, y)
    within = screening.screen_values(X_week, y)
    missing = screening.screen_missingness(X_raw, y)
    drift_rho = drift.sensor_drift(X)
    groups = correlated_groups(X)
    stability, mean_kept = model.stability_selection(X, y, n_rounds=args.rounds, seed=args.seed)
    shortlist = build_shortlist(values, within, stability, drift_rho, groups, args.q)
    cv = model.cross_validated_scores(X, y, seed=args.seed)
    time_auc, time_auc_std = model.time_only_score(y, seed=args.seed)
    holdout = model.time_holdout_scores(X, y, seed=args.seed)

    weekly.to_csv(out / "weekly_fail_rate.csv")
    values.to_csv(out / "screening_values.csv", index=False)
    within.to_csv(out / "screening_within_week.csv", index=False)
    missing.to_csv(out / "screening_missingness.csv", index=False)
    stability.sort_values(ascending=False).to_csv(out / "stability.csv", header=True)
    shortlist.to_csv(out / "shortlist.csv", index=False)
    cv.to_csv(out / "model_cv.csv")
    holdout.to_csv(out / "model_time_holdout.csv")

    raw_hits = shortlist[shortlist["q_value"] < args.q]
    survivors = shortlist[shortlist["survives"]]
    plots.p_chart(weekly, out / "p_chart.png")
    plots.volcano(values, set(raw_hits["sensor"]), set(survivors["sensor"]), out / "volcano.png")
    plots.stability(stability, out / "stability.png")
    leaders = survivors.loc[survivors["best_in_group"], "sensor"].tolist()
    if leaders:
        plots.top_sensors(X_week, y, leaders, out / "top_sensors.png")

    pd.set_option("display.width", 180)
    n_tested = int(values["p_value"].notna().sum())
    print(f"\nfail rate: first half of runs {halves['first_half']:.1%}, second half "
          f"{halves['second_half']:.1%} (Fisher p = {halves['p_value']:.2g})")
    print(f"weeks out of control: {int(weekly['out_of_control'].sum())} of {len(weekly)}")
    print(f"\nraw p < 0.05: {int((values['p_value'] < 0.05).sum())} sensors "
          f"(about {0.05 * n_tested:.0f} expected by chance)")
    print(f"raw values, q < {args.q:g}: {len(raw_hits)} sensors in {raw_hits['group'].nunique()} groups, "
          f"{int(raw_hits['drifting'].sum())} of them drifting")
    print(f"within-week values, q < {args.q:g}: {len(survivors)} sensors in {survivors['group'].nunique()} groups "
          f"({int((survivors['q_value'] < args.q).sum())} were also raw hits)")
    print(f"missingness linked to fails, q < {args.q:g}: {int((missing['q_value'] < args.q).sum())} sensors")
    print(f"stability selection: {mean_kept:.1f} sensors kept per round on average")
    print("\nSensors that survive the within-week comparison:")
    print(survivors.drop(columns="survives").round(4).to_string(index=False))
    print(f"\nRandom 5x3 CV (chance: ROC-AUC 0.5, PR-AUC {y.mean():.3f}):")
    print(cv.round(3).to_string())
    print(f"run order only     {time_auc:.3f}        {time_auc_std:.3f}   <- no sensor data at all")
    a = holdout.attrs
    print(f"\nTime holdout, first {a['n_train']} runs -> last {a['n_test']} "
          f"({a['fails_test']} fails, chance PR-AUC {a['test_fail_rate']:.3f}):")
    print(holdout.round(3).to_string())
    print(f"\nwrote tables and figures to {out}/")


if __name__ == "__main__":
    main()
