from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from sklearn.cross_decomposition import CCA

from .config import SPN_NAMES
from .io import read_csv


STATE_POSITIONS = {
    0: (0.00, 0.00),
    8: (2.55, 1.58),
    2: (2.55, 0.55),
    1: (2.55, -0.42),
    4: (2.55, -1.48),
    10: (5.00, 1.58),
    9: (5.00, 0.80),
    3: (5.00, 0.12),
    6: (5.00, -0.62),
    5: (5.00, -1.48),
    11: (7.35, 1.18),
    7: (7.35, -1.05),
    12: (9.30, 0.00),
}

DISPLAY_STATES = {
    "CBGT": [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12],
    "IBL": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Steinmetz": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
}

LOW_FREQUENCY_STATES = {
    "CBGT": {7, 11},
    "IBL": {7, 11},
    "Steinmetz": {7},
}

# Long outer-path transitions are routed around the intervening states, as in
# the manuscript CLAW diagrams.  Reverse directions use the opposite bend.
SPECIAL_EDGE_RADII = {
    (8, 12): -0.20,
    (12, 8): 0.20,
    (4, 12): 0.20,
    (12, 4): -0.20,
}

F_FEATURE_NAMES = [
    "FSI",
    "CxI",
    "GPi_sum",
    "STN_sum",
    "GPeP_sum",
    "GPeA_sum",
    "dSPN_sum",
    "iSPN_sum",
    "Cx_sum",
    "Th_sum",
    "GPi_diff",
    "STN_diff",
    "GPeP_diff",
    "GPeA_diff",
    "dSPN_diff",
    "iSPN_diff",
    "Cx_diff",
    "Th_diff",
]

DDM_PARAMETER_NAMES = ["a", "v", "t", "z"]
CONTROL_ENSEMBLE_NAMES = ["choice", "responsiveness", "pliancy"]


def state_bits(state: int) -> list[int]:
    return [int(value) for value in f"{int(state):04b}"]


def state_bit_string(state: int) -> str:
    return "[" + ",".join(map(str, state_bits(state))) + "]"


def compress_repeats(values: Iterable[int]) -> list[int]:
    compressed: list[int] = []
    for value in values:
        value = int(value)
        if not compressed or compressed[-1] != value:
            compressed.append(value)
    return compressed


def firing_threshold(
    values: np.ndarray,
    quantile: float = 0.90,
    bin_count: int = 100,
    bandwidth: float = 3.0,
    bimodal_separation: float = 0.65,
) -> float:
    """Return the empirical CLAW threshold used in the source notebooks."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    if np.ptp(values) == 0:
        return float(values[0])

    counts, bin_edges = np.histogram(values, bins=bin_count)
    kde = gaussian_kde(values, bw_method=bandwidth)
    kde_x = np.linspace(values.min(), values.max(), 1000)
    peaks = argrelextrema(kde(kde_x), np.greater)[0]

    if len(peaks) > 1:
        first, second = peaks[:2]
        separation = (kde_x[second] - kde_x[first]) / (
            kde_x.max() - kde_x.min() + 1e-8
        )
        if separation < bimodal_separation:
            return float((kde_x[first] + kde_x[second]) / 2.0)

    cumulative_frequency = np.cumsum(counts) / (len(values) + 1e-8)
    index = int(np.argmax(cumulative_frequency > quantile))
    return float(bin_edges[index])


def binarize_activity(activity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Binarize empirical population activity separately within each session."""
    tables: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []

    for session, session_df in activity.groupby("session", sort=False):
        thresholds: dict[str, float] = {}
        for base, members in {
            "dSPN": ["dSPN_left", "dSPN_right"],
            "iSPN": ["iSPN_left", "iSPN_right"],
        }.items():
            pooled = session_df[members].to_numpy(dtype=float).reshape(-1)
            pooled = pooled[np.isfinite(pooled)]
            nonzero = pooled[pooled > 0]
            if nonzero.size:
                pooled = nonzero
            threshold = firing_threshold(pooled)
            thresholds[base] = threshold
            threshold_rows.append(
                {
                    "session": session,
                    "population": base,
                    "threshold": threshold,
                }
            )

        binary = session_df.copy()
        for name in SPN_NAMES:
            base = name.rsplit("_", 1)[0]
            binary[name] = (binary[name] > thresholds[base]).astype(int)
        binary["state"] = sum(
            binary[name] * (2 ** (3 - index))
            for index, name in enumerate(SPN_NAMES)
        ).astype(int)
        tables.append(binary)

    return pd.concat(tables, ignore_index=True), pd.DataFrame(threshold_rows)


def trial_sequences(state_bins: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["dataset", "session", "trial_id"]
    ordered = state_bins.sort_values(group_columns + ["bin_index"])

    for key, trial in ordered.groupby(group_columns, sort=False):
        sequence = compress_repeats(trial["state"].to_numpy())
        if not sequence:
            continue
        rows.append(
            {
                "dataset": key[0],
                "session": key[1],
                "trial_id": key[2],
                "choice_left": int(trial["choice_left"].iloc[0]),
                "decision_time_ms": float(trial["decision_time_ms"].iloc[0]),
                "sequence": " ".join(map(str, sequence)),
                "ending_state": int(sequence[-1]),
            }
        )
    return pd.DataFrame(rows)


def parse_sequence(text: str) -> list[int]:
    return [int(value) for value in str(text).split()]


def _choice_balance_weights(trials: pd.DataFrame) -> pd.Series:
    weights = pd.Series(index=trials.index, dtype=float)
    for _session, group in trials.groupby("session", sort=False):
        p_left = float(group["choice_left"].mean())
        weights.loc[group.index[group["choice_left"] == 1]] = 0.5 / (p_left + 1e-8)
        weights.loc[group.index[group["choice_left"] == 0]] = 0.5 / (1.0 - p_left + 1e-8)
    return weights


def summarize_claw(
    trials: pd.DataFrame,
    balance_choice: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize state visits, terminality, and outgoing transition probabilities."""
    trials = trials.copy()
    trials["parsed_sequence"] = trials["sequence"].map(parse_sequence)
    trials["choice_weight"] = (
        _choice_balance_weights(trials) if balance_choice else 1.0
    )

    transition_counts: dict[int, Counter] = defaultdict(Counter)
    end_counts: Counter = Counter()
    state_trial_rows: list[dict[str, float | int]] = []

    for row in trials.itertuples(index=False):
        sequence = row.parsed_sequence
        for state in set(sequence):
            state_trial_rows.append(
                {
                    "state": int(state),
                    "choice_left": int(row.choice_left),
                    "decision_time_ms": float(row.decision_time_ms),
                    "choice_weight": float(row.choice_weight),
                }
            )
        for source, target in zip(sequence[:-1], sequence[1:]):
            transition_counts[int(source)][int(target)] += 1
        end_counts[int(sequence[-1])] += 1

    visits = pd.DataFrame(state_trial_rows)
    node_rows: list[dict[str, object]] = []
    for state, group in visits.groupby("state", sort=True):
        outgoing = sum(transition_counts[int(state)].values()) + end_counts[int(state)]
        node_rows.append(
            {
                "state": int(state),
                "bits": state_bit_string(int(state)),
                "n_trials": int(len(group)),
                "left_choice_probability": float(
                    np.average(group["choice_left"], weights=group["choice_weight"])
                ),
                "mean_decision_time_ms": float(group["decision_time_ms"].mean()),
                "median_decision_time_ms": float(group["decision_time_ms"].median()),
                "terminal_probability": float(end_counts[int(state)] / outgoing),
                "end_count": int(end_counts[int(state)]),
                "outgoing_total": int(outgoing),
            }
        )

    edge_rows: list[dict[str, object]] = []
    for source, targets in transition_counts.items():
        total = sum(targets.values()) + end_counts[source]
        for target, count in targets.items():
            edge_rows.append(
                {
                    "source": int(source),
                    "target": int(target),
                    "count": int(count),
                    "probability": float(count / total),
                }
            )

    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def transition_table_with_end(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Return outgoing transition probabilities, including END transitions."""
    rows: list[dict[str, object]] = []

    if not edges.empty:
        for row in edges.itertuples(index=False):
            source = int(row.source)
            target = int(row.target)
            rows.append(
                {
                    "source": source,
                    "source_bits": state_bit_string(source),
                    "target": target,
                    "target_bits": state_bit_string(target),
                    "transition_type": "state_to_state",
                    "count": int(row.count),
                    "probability": float(row.probability),
                }
            )

    for row in nodes.itertuples(index=False):
        probability = float(row.terminal_probability)
        if probability <= 0:
            continue
        count = int(row.end_count) if hasattr(row, "end_count") else np.nan
        rows.append(
            {
                "source": int(row.state),
                "source_bits": state_bit_string(int(row.state)),
                "target": "END",
                "target_bits": "END",
                "transition_type": "state_to_END",
                "count": count,
                "probability": probability,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    target_order = out["target"].map(lambda value: 99 if str(value) == "END" else int(value))
    return (
        out.assign(_target_order=target_order)
        .sort_values(["source", "transition_type", "_target_order"], ignore_index=True)
        .drop(columns="_target_order")
    )


def _strip_unnamed(table: pd.DataFrame) -> pd.DataFrame:
    return table.loc[:, ~table.columns.astype(str).str.contains(r"^Unnamed")].copy()


def _select_rows_by_index_or_position(table: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    try:
        return table.loc[indices].copy()
    except KeyError:
        return table.iloc[[index for index in indices if 0 <= index < len(table)]].copy()


def _f_vector(rows: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            rows["FSI_common"].mean(),
            rows["CxI_common"].mean(),
            (rows["GPi_left"] + rows["GPi_right"]).mean(),
            (rows["STN_left"] + rows["STN_right"]).mean(),
            (rows["GPeP_left"] + rows["GPeP_right"]).mean(),
            (rows["GPeA_left"] + rows["GPeA_right"]).mean(),
            (rows["dSPN_left"] + rows["dSPN_right"]).mean(),
            (rows["iSPN_left"] + rows["iSPN_right"]).mean(),
            (rows["Cx_left"] + rows["Cx_right"]).mean(),
            (rows["Th_left"] + rows["Th_right"]).mean(),
            (rows["GPi_left"] - rows["GPi_right"]).mean(),
            (rows["STN_left"] - rows["STN_right"]).mean(),
            (rows["GPeP_left"] - rows["GPeP_right"]).mean(),
            (rows["GPeA_left"] - rows["GPeA_right"]).mean(),
            (rows["dSPN_left"] - rows["dSPN_right"]).mean(),
            (rows["iSPN_left"] - rows["iSPN_right"]).mean(),
            (rows["Cx_left"] - rows["Cx_right"]).mean(),
            (rows["Th_left"] - rows["Th_right"]).mean(),
        ],
        dtype=float,
    )


def build_cbgt_state_activity(networks_dir: Path, n_networks: int = 300) -> pd.DataFrame:
    """Build the 18D network-by-state activity table from the uploaded network CSVs."""
    rows: list[dict[str, object]] = []
    networks_dir = Path(networks_dir)

    for network_id in range(1, n_networks + 1):
        network_dir = networks_dir / f"network_{network_id}"
        state_path = network_dir / "data_conf_collapsed.csv"
        if not state_path.exists():
            state_path = network_dir / "data_conf.csv"
        state_table = _strip_unnamed(read_csv(state_path))
        firing_table = _strip_unnamed(read_csv(network_dir / "binned_firing_rates.csv"))

        if "state" in state_table.columns:
            state_table["state4"] = state_table["state"].astype(int)
        elif "state4" not in state_table.columns:
            state_table["state4"] = sum(
                state_table[name].astype(int) * (2 ** (3 - index))
                for index, name in enumerate(SPN_NAMES)
            ).astype(int)

        phase_zero = (
            state_table.loc[state_table["phase"] == 0]
            if "phase" in state_table.columns
            else state_table
        )
        for state, group in phase_zero.groupby("state4", sort=True):
            selected = _select_rows_by_index_or_position(
                firing_table,
                [int(index) for index in group.index],
            )
            if selected.empty:
                continue
            vector = _f_vector(selected)
            row: dict[str, object] = {
                "network_id": int(network_id),
                "state": int(state),
                "bits": state_bit_string(int(state)),
                "n_bins": int(len(selected)),
            }
            row.update({name: float(value) for name, value in zip(F_FEATURE_NAMES, vector)})
            row.update(
                {
                    f"feature_{index:02d}": float(value)
                    for index, value in enumerate(vector)
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def fit_control_ensembles(
    f_matrix: np.ndarray,
    d_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the three CCA components used for choice, responsiveness, and pliancy."""
    firing = np.asarray(f_matrix, dtype=float)
    ddm = np.asarray(d_matrix, dtype=float)
    firing = (firing - firing.mean(axis=0)) / firing.std(axis=0)
    ddm = (ddm - ddm.mean(axis=0)) / ddm.std(axis=0)

    cca = CCA(n_components=3)
    cca.fit_transform(firing, ddm)
    neural_loadings = cca.x_loadings_.copy()
    ddm_loadings = cca.y_loadings_.copy()

    neural_loadings[:, 2] *= -1
    ddm_loadings[:, 2] *= -1
    return neural_loadings, ddm_loadings


def control_transition_scores(
    state_activity: pd.DataFrame,
    edges: pd.DataFrame,
    neural_loadings: np.ndarray,
    ddm_loadings: np.ndarray,
) -> pd.DataFrame:
    feature_columns = [
        f"feature_{index:02d}" for index in range(neural_loadings.shape[0])
    ]
    rows: list[dict[str, object]] = []

    for edge in edges.itertuples(index=False):
        control_changes: list[np.ndarray] = []
        ddm_changes: list[np.ndarray] = []

        for _network_id, network in state_activity.groupby("network_id", sort=False):
            means = network.groupby("state", sort=False)[feature_columns].mean()
            if int(edge.source) not in means.index or int(edge.target) not in means.index:
                continue
            delta = (
                means.loc[int(edge.target)].to_numpy(dtype=float)
                - means.loc[int(edge.source)].to_numpy(dtype=float)
            )
            control = delta @ neural_loadings
            ddm = control @ ddm_loadings.T
            control_changes.append(control)
            ddm_changes.append(ddm)

        if not control_changes:
            continue

        median_control = np.median(np.vstack(control_changes), axis=0)
        median_ddm = np.median(np.vstack(ddm_changes), axis=0)
        row: dict[str, object] = {
            "source": int(edge.source),
            "target": int(edge.target),
            "count": int(edge.count),
            "probability": float(edge.probability),
            "n_common_networks": int(len(control_changes)),
        }
        row.update(
            {
                name: float(value)
                for name, value in zip(CONTROL_ENSEMBLE_NAMES, median_control)
            }
        )
        row.update(
            {
                name: float(value)
                for name, value in zip(DDM_PARAMETER_NAMES, median_ddm)
            }
        )
        row.update({f"ddm_{index}": float(value) for index, value in enumerate(median_ddm)})
        rows.append(row)

    return pd.DataFrame(rows)
