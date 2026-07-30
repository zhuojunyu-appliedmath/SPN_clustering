from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .clustering import (
    _exact_corr_matrix,
    _stein_cluster_vectors_exact,
    _stein_plot_trial_info_exact,
    _stein_split_fast_slow_exact,
    cluster_steinmetz_session_exact,
)
from .datasets import load_steinmetz_session_exact


STEINMETZ_SESSION_ORDER = (
    "Hench_2017-06-18",
    "Lederberg_2017-12-11",
    "Radnitz_2017-01-12",
    "Richards_2017-11-01",
)

STEINMETZ_SESSION_LABELS = {
    "Hench_2017-06-18": "Hench | 2017-06-18",
    "Lederberg_2017-12-11": "Lederberg | 2017-12-11",
    "Radnitz_2017-01-12": "Radnitz | 2017-01-12",
    "Richards_2017-11-01": "Richards | 2017-11-01",
}


def _rng_for_key(key: str, base_seed: int = 0) -> np.random.Generator:
    """Return the MD5-keyed RNG used in the source ISI notebook."""
    digest = hashlib.md5(key.encode("utf-8")).digest()
    local_seed = (int.from_bytes(digest[:4], "little") + int(base_seed)) % 2**32
    return np.random.default_rng(local_seed)


def shuffle_isi_train(
    spike_times: np.ndarray,
    rng: np.random.Generator,
    keep_first: bool = True,
) -> np.ndarray:
    """Permute one unit's ISIs while preserving spike count and ISI multiset."""
    times = np.asarray(spike_times, dtype=float).reshape(-1)
    if times.size <= 2:
        return times.copy()

    times = np.sort(times)
    shuffled_isi = rng.permutation(np.diff(times))
    shuffled = np.empty_like(times)
    shuffled[0] = times[0] if keep_first else 0.0
    shuffled[1:] = shuffled[0] + np.cumsum(shuffled_isi)
    return shuffled


def _isi_shuffled_payload(
    payload: dict[str, Any],
    session_seed: int,
    repetition: int,
    shuffle_seed_offset: int = 123456,
) -> dict[str, Any]:
    """Replace target-unit spike trains with the source notebook's ISI shuffles."""
    spike_times = np.asarray(payload["spike_times"], dtype=float)
    spike_clusters = np.asarray(payload["spike_clusters"], dtype=int)
    session = str(payload["session"])

    shuffled_times: list[np.ndarray] = []
    shuffled_clusters: list[np.ndarray] = []

    for unit_id in np.asarray(payload["unit_ids_all"], dtype=int):
        unit_times = np.sort(spike_times[spike_clusters == int(unit_id)])
        rng = _rng_for_key(
            f"{session}|rep={int(repetition)}|uid={int(unit_id)}",
            base_seed=int(session_seed) + int(shuffle_seed_offset),
        )
        unit_shuffled = shuffle_isi_train(unit_times, rng=rng, keep_first=True)
        shuffled_times.append(unit_shuffled)
        shuffled_clusters.append(
            np.full(unit_shuffled.shape, int(unit_id), dtype=int)
        )

    times = np.concatenate(shuffled_times)
    clusters = np.concatenate(shuffled_clusters)
    order = np.argsort(times)

    shuffled_payload = dict(payload)
    shuffled_payload["spike_times"] = times[order]
    shuffled_payload["spike_clusters"] = clusters[order]
    return shuffled_payload


def _rename_from_unshuffled_spikes(
    result: dict[str, Any],
    original_payload: dict[str, Any],
    matching_seed: int,
) -> dict[str, Any]:

    renamed = dict(result)
    renamed["source_payload"] = original_payload

    labels = np.asarray(renamed["final_labels"], dtype=int)
    names = np.array(["unassigned"] * len(labels), dtype=object)
    names[labels == 0] = "L_sub0"
    names[labels == 1] = "L_sub1"
    names[labels == 2] = "R_sub0"
    names[labels == 3] = "R_sub1"

    trial_info = _stein_plot_trial_info_exact(
        original_payload,
        matching_seed=int(matching_seed),
    )
    fast_trials, slow_trials, split_metadata = _stein_split_fast_slow_exact(
        trial_info["trials_all"],
        trial_info["reaction_time_s"],
    )

    corr_fast = _exact_corr_matrix(
        _stein_cluster_vectors_exact(renamed, fast_trials)
    )
    corr_slow = _exact_corr_matrix(
        _stein_cluster_vectors_exact(renamed, slow_trials)
    )
    delta = corr_slow - corr_fast

    best_pair: tuple[int, int] | None = None
    best_drop = np.inf
    for left_label in (0, 1):
        for right_label in (2, 3):
            drop = delta[left_label, right_label]
            if np.isfinite(drop) and drop < best_drop:
                best_drop = float(drop)
                best_pair = (left_label, right_label)

    name_map: dict[int, str] = {}
    if best_pair is not None:
        d_left, d_right = best_pair
        name_map = {
            int(d_left): "dSPN_left",
            int(1 - d_left): "iSPN_left",
            int(d_right): "dSPN_right",
            int(5 - d_right): "iSPN_right",
        }
        for label, name in name_map.items():
            names[labels == label] = name

    renamed.update(
        final_names=names,
        name_map=name_map,
        corr_fast=corr_fast,
        corr_slow=corr_slow,
        corr_drop=float(best_drop) if best_pair is not None else np.nan,
        fast_trials=fast_trials,
        slow_trials=slow_trials,
        split_metadata=split_metadata,
        naming_trial_info=trial_info,
    )
    return renamed


def extract_early_mid_late_bar_means(
    result: dict[str, Any],
) -> np.ndarray | None:

    required = ("dSPN_left", "iSPN_left", "dSPN_right", "iSPN_right")
    x12 = np.asarray(result["x12_z"], dtype=float)
    labels = np.asarray(result["final_labels"], dtype=int)
    names = np.asarray(result["final_names"], dtype=object)

    assigned = labels >= 0
    x12 = x12[assigned]
    labels = labels[assigned]
    names = names[assigned]

    label_to_name: dict[int, str] = {}
    for label in range(4):
        rows = np.where(labels == label)[0]
        if rows.size:
            label_to_name[label] = str(names[rows[0]])

    if not all(name in label_to_name.values() for name in required):
        return None

    name_to_label = {name: label for label, name in label_to_name.items()}
    output = np.full((2, 3, 4), np.nan, dtype=float)
    segment_bins = ((0, 1), (2, 3), (4, 5))

    for side_index, offset in enumerate((0, 6)):
        for segment_index, pair in enumerate(segment_bins):
            columns = [offset + pair[0], offset + pair[1]]
            for subtype_index, name in enumerate(required):
                label = int(name_to_label[name])
                rows = np.where(labels == label)[0]
                values = x12[rows][:, columns].mean(axis=1)
                output[side_index, segment_index, subtype_index] = float(
                    np.mean(values)
                )
    return output


def bar_pattern_scores(
    bar_means: np.ndarray,
    late_segment: int = 2,
) -> dict[str, float]:

    d_left_on_left = bar_means[0, :, 0]
    i_left_on_left = bar_means[0, :, 1]
    d_right_on_left = bar_means[0, :, 2]
    i_right_on_left = bar_means[0, :, 3]

    d_left_on_right = bar_means[1, :, 0]
    i_left_on_right = bar_means[1, :, 1]
    d_right_on_right = bar_means[1, :, 2]
    i_right_on_right = bar_means[1, :, 3]

    left_dominance = (
        d_left_on_left + i_left_on_left
    ) - (
        d_right_on_left + i_right_on_left
    )
    right_dominance = (
        d_right_on_right + i_right_on_right
    ) - (
        d_left_on_right + i_left_on_right
    )

    scores = {
        "chan_dom_L_min": float(np.min(left_dominance)),
        "chan_dom_L_mean": float(np.mean(left_dominance)),
        "chan_dom_R_min": float(np.min(right_dominance)),
        "chan_dom_R_mean": float(np.mean(right_dominance)),
        "late_L_d_minus_i": float(
            d_left_on_left[late_segment] - i_left_on_left[late_segment]
        ),
        "late_L_i_minus_dR": float(
            i_right_on_left[late_segment] - d_right_on_left[late_segment]
        ),
        "late_R_d_minus_i": float(
            d_right_on_right[late_segment] - i_right_on_right[late_segment]
        ),
        "late_R_i_minus_dL": float(
            i_left_on_right[late_segment] - d_left_on_right[late_segment]
        ),
    }
    scores["min_required_margin"] = float(
        min(
            scores["chan_dom_L_min"],
            scores["chan_dom_R_min"],
            scores["late_L_d_minus_i"],
            scores["late_L_i_minus_dR"],
            scores["late_R_d_minus_i"],
            scores["late_R_i_minus_dL"],
        )
    )
    return scores


def strict_bar_pattern_pass(
    bar_means: np.ndarray | None,
    fraction_threshold: float = 1.0,
    margin: float = 0.0,
    late_segment: int = 2,
) -> bool:
    """Apply the six inequalities defining a pattern pass."""
    if bar_means is None:
        return False

    d_left_on_left = bar_means[0, :, 0]
    i_left_on_left = bar_means[0, :, 1]
    d_right_on_left = bar_means[0, :, 2]
    i_right_on_left = bar_means[0, :, 3]

    d_left_on_right = bar_means[1, :, 0]
    i_left_on_right = bar_means[1, :, 1]
    d_right_on_right = bar_means[1, :, 2]
    i_right_on_right = bar_means[1, :, 3]

    left_series = np.mean(
        (
            (d_left_on_left + i_left_on_left)
            - (d_right_on_left + i_right_on_left)
        )
        > margin
    ) >= fraction_threshold
    left_direct = (
        d_left_on_left[late_segment] - i_left_on_left[late_segment]
    ) > margin
    left_indirect = (
        i_right_on_left[late_segment] - d_right_on_left[late_segment]
    ) > margin

    right_series = np.mean(
        (
            (d_right_on_right + i_right_on_right)
            - (d_left_on_right + i_left_on_right)
        )
        > margin
    ) >= fraction_threshold
    right_direct = (
        d_right_on_right[late_segment] - i_right_on_right[late_segment]
    ) > margin
    right_indirect = (
        i_left_on_right[late_segment] - d_left_on_right[late_segment]
    ) > margin

    return bool(
        left_series
        and left_direct
        and left_indirect
        and right_series
        and right_direct
        and right_indirect
    )


def run_steinmetz_isi_shuffle_session(
    session_dir: Path,
    session_name: str,
    matching_seed: int,
    premove_gap_s: float = 0.010,
    n_shuffles: int = 50,
) -> pd.DataFrame:

    original_payload = load_steinmetz_session_exact(
        Path(session_dir),
        session_name=str(session_name),
        matching_seed=int(matching_seed),
        premove_gap_s=float(premove_gap_s),
    )

    records: list[dict[str, Any]] = []
    for repetition in range(int(n_shuffles)):
        record: dict[str, Any] = {
            "session": str(session_name),
            "matching_seed": int(matching_seed),
            "repetition": int(repetition),
            "pipeline_success": False,
            "pattern_pass": False,
        }
        try:
            shuffled_payload = _isi_shuffled_payload(
                original_payload,
                session_seed=int(matching_seed),
                repetition=int(repetition),
            )
            result = cluster_steinmetz_session_exact(shuffled_payload)
            result = _rename_from_unshuffled_spikes(
                result,
                original_payload=original_payload,
                matching_seed=int(matching_seed),
            )

            bar_means = extract_early_mid_late_bar_means(result)
            record["pipeline_success"] = True
            record["pattern_pass"] = strict_bar_pattern_pass(
                bar_means,
                fraction_threshold=1.0,
                margin=0.0,
                late_segment=2,
            )
            if bar_means is not None:
                record.update(bar_pattern_scores(bar_means, late_segment=2))
        except Exception as error:
            record["error"] = str(error)

        records.append(record)

    return pd.DataFrame(records)


def summarize_isi_shuffle_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize pattern passes using all repetitions, including failed pipelines."""
    return (
        runs.groupby("session", as_index=False)
        .agg(
            matching_seed=("matching_seed", "first"),
            n_shuffles=("repetition", "size"),
            n_pipeline_success=("pipeline_success", "sum"),
            false_positive_count=("pattern_pass", "sum"),
            false_positive_rate=("pattern_pass", "mean"),
        )
    )


def plot_isi_false_positive_rates(
    summary: pd.DataFrame,
    session_order: tuple[str, ...] = STEINMETZ_SESSION_ORDER,
):
    """Plot the manuscript SI Fig. S2(B) horizontal-bar panel."""
    data = summary.set_index("session").reindex(session_order).reset_index()
    values = data["false_positive_rate"].to_numpy(float)
    labels = [STEINMETZ_SESSION_LABELS[session] for session in data["session"]]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    y_positions = np.arange(len(data))
    bars = ax.barh(y_positions, values, height=0.4)
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0.0, 0.10)
    ax.set_xticks(np.arange(0.0, 0.101, 0.02))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("False positive rates (pattern pass rates)")
    ax.set_title("(B) ISI-shuffle false positive rates")

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
        )

    fig.tight_layout()
    return fig, ax
