import numpy as np
import pandas as pd
import pytest

from secom import drift, model, screening
from secom.data import clean, correlated_groups, load_secom
from secom.report import build_shortlist
from secom.synthetic import make_secom, write_secom


@pytest.fixture(scope="module")
def synth():
    X, y, t, truth = make_secom(seed=1)
    X_clean, summary = clean(X)
    return X, X_clean, y, t, truth, summary


def test_files_round_trip(tmp_path, synth):
    X, _, y, t, _, _ = synth
    write_secom(tmp_path, X, y, t)
    X2, y2, t2 = load_secom(tmp_path)
    assert X2.shape == X.shape and list(X2.columns) == list(X.columns)
    assert (y2 == y).all() and t2.is_monotonic_increasing
    assert np.allclose(X2.to_numpy(), X.to_numpy(), equal_nan=True)


def test_loader_sorts_by_time(tmp_path):
    X = pd.DataFrame({"s000": [1.0, 2.0, 3.0], "s001": [4.0, np.nan, 6.0]})
    t = pd.Series(pd.to_datetime(["2008-08-03", "2008-08-01", "2008-08-02"]))
    write_secom(tmp_path, X, pd.Series([1, 0, 0]), t)
    X2, y2, _ = load_secom(tmp_path)
    assert X2["s000"].tolist() == [2.0, 3.0, 1.0] and y2.tolist() == [0, 0, 1]


def test_clean_drops_constant_and_mostly_missing(synth):
    _, X_clean, _, _, truth, summary = synth
    assert summary["dropped_constant"] == len(truth["constant"])
    assert summary["dropped_missing"] == len(truth["mostly_missing"])
    assert not set(truth["constant"] + truth["mostly_missing"]) & set(X_clean.columns)


def test_screening_recovers_planted_sensors_and_controls_false_discoveries(synth):
    _, X_clean, y, _, truth, _ = synth
    table = screening.screen_values(X_clean, y)
    hits = set(table.loc[table["q_value"] < 0.10, "sensor"])
    real = set(truth["signal"] + truth["copies"])
    assert len(hits & real) >= 12                      # 14 planted
    assert len(hits - real) <= 0.3 * len(hits)         # FDR target is 10%
    # raw p-values, by contrast, flag many unrelated sensors
    assert (table["p_value"] < 0.05).sum() > len(hits)


def test_missingness_screen_finds_dropout_sensors(synth):
    X, _, y, _, truth, _ = synth
    table = screening.screen_missingness(X, y)
    hits = set(table.loc[table["q_value"] < 0.10, "sensor"])
    planted = set(truth["informative_missing"])
    assert planted <= hits and len(hits - planted) <= 1   # FDR allows the odd false hit
    assert set(table["sensor"].head(2)) == planted        # and the planted ones rank first


def test_copies_are_grouped_with_their_original(synth):
    _, X_clean, _, _, truth, _ = synth
    groups = correlated_groups(X_clean)
    for i, copy in enumerate(truth["copies"]):
        assert groups[copy] == groups[truth["signal"][i // 2]]
    assert groups.nunique() == X_clean.shape[1] - len(truth["copies"])


def test_p_chart_flags_only_the_excursion(synth):
    _, _, y, t, truth, _ = synth
    weekly = drift.weekly_fail_rate(y, t)
    assert weekly["fails"].sum() == y.sum() and weekly["n"].sum() == len(y)
    flagged = ((weekly.index[weekly["out_of_control"]] - pd.Timestamp("2008-07-19")).days // 7).tolist()
    assert flagged and set(flagged) <= {5, 6, 7}       # calendar weeks straddle the planted ones


def test_drift_sensors_are_flagged(synth):
    _, X_clean, _, _, truth, _ = synth
    rho = drift.sensor_drift(X_clean).abs()
    assert rho[truth["drift"]].min() > 0.3 > rho.drop(truth["drift"]).max()


def test_stability_and_shortlist(synth):
    _, X_clean, y, _, truth, _ = synth
    freq, mean_kept = model.stability_selection(X_clean, y, n_rounds=20)
    assert 0 < mean_kept < 80
    assert freq[truth["signal"]].mean() > 0.5 > freq.drop(truth["signal"] + truth["copies"]).mean()

    _, _, _, t, _, _ = synth
    within = screening.screen_values(drift.center_by_week(X_clean, t), y)
    shortlist = build_shortlist(screening.screen_values(X_clean, y), within, freq,
                                drift.sensor_drift(X_clean), correlated_groups(X_clean), 0.10)
    survivors = shortlist[shortlist["survives"]]
    leaders = set(survivors.loc[survivors["best_in_group"], "sensor"])
    # planted causes act within a week, so they must survive the time adjustment
    assert len(leaders & set(truth["signal"] + truth["copies"])) >= 7   # 8 independent causes
    assert shortlist["sensor"].is_unique


def test_within_week_screen_rejects_a_pure_time_confound():
    """Fail rate falls over time and a sensor rises over time, with no link inside a week."""
    rng = np.random.default_rng(0)
    n = 1500
    t = pd.Series(pd.Timestamp("2008-07-21") + pd.to_timedelta(np.sort(rng.uniform(0, 70, n)), unit="D"))
    progress = np.linspace(0, 1, n)
    week_index = (t - t.min()).dt.days // 7
    y = pd.Series((rng.random(n) < 0.20 - 0.018 * week_index).astype(int))
    X = pd.DataFrame({"trend": week_index + rng.standard_normal(n),          # steps week to week
                      "cause": rng.standard_normal(n) + 1.0 * y,              # real within-week effect
                      "noise": rng.standard_normal(n)})
    raw = screening.screen_values(X, y).set_index("sensor")["p_value"]
    adj = screening.screen_values(drift.center_by_week(X, t), y).set_index("sensor")["p_value"]
    assert raw["trend"] < 1e-4 and adj["trend"] > 0.01      # confound: significant only before adjusting
    assert raw["cause"] < 1e-6 and adj["cause"] < 1e-6      # real cause: significant both ways
    assert progress[0] == 0 and model.time_only_score(y)[0] > 0.6   # time alone predicts fails here


def test_center_by_week_removes_weekly_medians(synth):
    _, X_clean, _, t, _, _ = synth
    centred = drift.center_by_week(X_clean, t)
    week = t.dt.to_period("W").astype(str).to_numpy()
    assert centred.groupby(week).median().abs().max().max() < 1e-9
    assert centred.isna().equals(X_clean.isna())


def test_models_beat_chance_on_planted_signal(synth):
    _, X_clean, y, _, _, _ = synth
    holdout = model.time_holdout_scores(X_clean, y)
    assert (holdout["roc_auc"] > 0.75).all()
    assert holdout.attrs["n_train"] + holdout.attrs["n_test"] == len(y)
