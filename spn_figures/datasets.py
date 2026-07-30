from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .config import (
    BASELINE_ADD_HZ,
    BASELINE_WINDOW_S,
    BIN_SIZE_S,
    LAST_K,
    SMOOTH_SIGMA_S,
    SPN_NAMES,
)
from .io import read_csv


def causal_half_gaussian_kernel(sigma_s: float = SMOOTH_SIGMA_S, bin_size_s: float = BIN_SIZE_S) -> np.ndarray:
    sigma_bins = sigma_s / bin_size_s
    x = np.arange(int(np.ceil(4 * sigma_bins)) + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return kernel / kernel.sum()


def deterministic_rng(text: str, seed: int = 20) -> np.random.Generator:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return np.random.default_rng((int.from_bytes(digest[:4], "little") + seed) % 2**32)


def match_choice_trials(
    choice_left: np.ndarray,
    signed_contrast: np.ndarray,
    n_bins: int = 3,
    minimum_per_bin: int = 5,
    seed_text: str = "session",
    seed: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    rng = deterministic_rng(seed_text, seed=seed)
    choice_left = np.asarray(choice_left, dtype=bool)
    contrast = np.nan_to_num(np.asarray(signed_contrast, dtype=float))
    left = np.flatnonzero(choice_left)
    right = np.flatnonzero(~choice_left)
    edges = np.unique(np.quantile(contrast, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return left, right

    left_keep: list[int] = []
    right_keep: list[int] = []
    for edge_i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        upper = contrast <= high if edge_i == len(edges) - 2 else contrast < high
        in_bin = (contrast >= low) & upper
        left_bin = left[in_bin[left]]
        right_bin = right[in_bin[right]]
        n = min(len(left_bin), len(right_bin))
        if n >= minimum_per_bin:
            left_keep.extend(rng.choice(left_bin, n, replace=False).tolist())
            right_keep.extend(rng.choice(right_bin, n, replace=False).tolist())

    if not left_keep:
        return left, right
    return np.sort(left_keep), np.sort(right_keep)


def match_evidence_trials(
    choice_left: np.ndarray,
    evidence_magnitude: np.ndarray,
    n_bins: int = 3,
    minimum_per_bin: int = 5,
    seed_text: str = "session",
    seed: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    rng = deterministic_rng(seed_text, seed=seed)
    choice_left = np.asarray(choice_left, dtype=bool)
    evidence = np.asarray(evidence_magnitude, dtype=float)
    left = np.flatnonzero(choice_left & np.isfinite(evidence))
    right = np.flatnonzero((~choice_left) & np.isfinite(evidence))
    values = np.unique(evidence[np.isfinite(evidence)])
    if len(values) < n_bins:
        return np.flatnonzero(choice_left), np.flatnonzero(~choice_left)

    left_keep: list[int] = []
    right_keep: list[int] = []
    for group in np.array_split(np.sort(values), n_bins):
        low, high = float(group.min()), float(group.max())
        left_bin = left[(evidence[left] >= low) & (evidence[left] <= high)]
        right_bin = right[(evidence[right] >= low) & (evidence[right] <= high)]
        n = min(len(left_bin), len(right_bin))
        if n >= minimum_per_bin:
            left_keep.extend(rng.choice(left_bin, n, replace=False).tolist())
            right_keep.extend(rng.choice(right_bin, n, replace=False).tolist())

    if not left_keep:
        return np.flatnonzero(choice_left), np.flatnonzero(~choice_left)
    return np.sort(left_keep), np.sort(right_keep)



def _strip_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")].copy()


def _cbgt_trial_keys(df: pd.DataFrame) -> list[str]:
    return [col for col in ("seed", "trial_num") if col in df.columns]


def _standardize_cbgt_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the original CBGT CSV columns to the shared notebook schema."""
    df = _strip_unnamed(df)
    keys = _cbgt_trial_keys(df)
    sort_cols = keys + (["bin_num"] if "bin_num" in df.columns else [])
    if sort_cols:
        df = df.sort_values(sort_cols).copy()

    df["trial_id"] = df.groupby(keys, sort=True).ngroup() + 1
    df["bin_index"] = (
        pd.to_numeric(df["bin_num"], errors="coerce").astype(int)
        if "bin_num" in df.columns
        else df.groupby(keys, sort=False).cumcount()
    )
    choice = df["chosen_action"].astype(str).str.lower()
    df["choice_left"] = choice.eq("left").astype(int)

    phase0 = df["phase"].eq(0)
    decision_time = (
        df.loc[phase0]
        .groupby("trial_id", sort=False)
        .size()
        .mul(BIN_SIZE_S * 1000.0)
    )
    df["decision_time_ms"] = df["trial_id"].map(decision_time)
    return df


def build_cbgt_manifest(networks_dir: Path, index_dir: Path) -> pd.DataFrame:
    """Build the 300-network manifest from the original 0-based speed-index pickles."""
    import pickle

    groups = {
        "fast": pickle.load(open(Path(index_dir) / "fast_indices.pickle", "rb")),
        "intermediate": pickle.load(open(Path(index_dir) / "medium_indices.pickle", "rb")),
        "slow": pickle.load(open(Path(index_dir) / "slow_indices.pickle", "rb")),
    }
    speed_by_network = {
        int(index) + 1: speed
        for speed, indices in groups.items()
        for index in indices
    }

    rows = []
    for network_id in range(1, 301):
        path = Path(networks_dir) / f"network_{network_id}" / "binned_firing_rates.csv"
        df = _standardize_cbgt_rows(read_csv(path)).query("phase == 0")
        median_dt = df.groupby("trial_id")["decision_time_ms"].first().median()
        rows.append(
            {
                "network_id": network_id,
                "median_decision_time_ms": float(median_dt),
                "speed_group": speed_by_network[network_id],
            }
        )
    return pd.DataFrame(rows)


def prepare_cbgt_tables(
    network_manifest: Path,
    networks_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    manifest = read_csv(network_manifest)
    feature_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    activity_rows: list[dict[str, Any]] = []
    predecision_rows: list[dict[str, Any]] = []

    for row in manifest.itertuples(index=False):
        network_id = int(row.network_id)
        speed_group = str(row.speed_group)
        path = Path(networks_dir) / f"network_{network_id}" / "binned_firing_rates.csv"
        df = _standardize_cbgt_rows(read_csv(path)).query("phase == 0")

        valid_trials: list[dict[str, Any]] = []

        for trial_id, trial in df.groupby("trial_id", sort=True):
            trial = trial.sort_values("bin_index")
            choice_left = int(trial["choice_left"].iloc[0])
            decision_time_ms = float(trial["decision_time_ms"].iloc[0])

            # SI Fig. S1D uses all pre-decision firing-rate rows, including the
            # occasional trial that is shorter than six bins.
            for name in SPN_NAMES:
                for bin_index, value in zip(
                    trial["bin_index"].to_numpy(int),
                    trial[name].to_numpy(float),
                ):
                    predecision_rows.append(
                        {
                            "network_id": network_id,
                            "speed_group": speed_group,
                            "trial_id": int(trial_id),
                            "choice_left": choice_left,
                            "decision_time_ms": decision_time_ms,
                            "spn_name": name,
                            "bin_index": int(bin_index),
                            "rate": float(value),
                        }
                    )

            # The clustering features require exactly the final six bins.
            if len(trial) < LAST_K:
                continue

            segment = trial.tail(LAST_K)
            traces = {
                name: segment[name].to_numpy(float)
                for name in SPN_NAMES
            }
            if any(
                len(values) != LAST_K or not np.all(np.isfinite(values))
                for values in traces.values()
            ):
                continue

            valid_trials.append(
                {
                    "trial_id": int(trial_id),
                    "choice_left": choice_left,
                    "decision_time_ms": decision_time_ms,
                    "traces": traces,
                }
            )

        # A 12D profile needs at least one retained trial for each choice.
        retained_choices = {trial["choice_left"] for trial in valid_trials}
        if retained_choices != {0, 1}:
            continue

        for trial in valid_trials:
            for name in SPN_NAMES:
                values = trial["traces"][name]
                for bin_from_decision, value in zip(range(-LAST_K, 0), values):
                    activity_rows.append(
                        {
                            "network_id": network_id,
                            "speed_group": speed_group,
                            "trial_id": trial["trial_id"],
                            "choice_left": trial["choice_left"],
                            "decision_time_ms": trial["decision_time_ms"],
                            "spn_name": name,
                            "bin_from_decision": bin_from_decision,
                            "rate": float(value),
                        }
                    )

        for name in SPN_NAMES:
            left_matrix = np.vstack(
                [trial["traces"][name] for trial in valid_trials if trial["choice_left"] == 1]
            )
            right_matrix = np.vstack(
                [trial["traces"][name] for trial in valid_trials if trial["choice_left"] == 0]
            )
            feature = np.concatenate(
                [left_matrix.mean(axis=0), right_matrix.mean(axis=0)]
            )

            unit_key = f"{network_id:03d}:{name}"
            feature_rows.append(
                {
                    "unit_key": unit_key,
                    "network_id": network_id,
                    "speed_group": speed_group,
                    "true_name": name,
                    "n_retained_trials": len(valid_trials),
                    **{f"feature_{i:02d}": float(value) for i, value in enumerate(feature)},
                }
            )

            vector = np.concatenate(
                [trial["traces"][name] for trial in valid_trials]
            )
            vector = (vector - vector.mean()) / (vector.std() + 1e-9)
            vector_rows.extend(
                {
                    "unit_key": unit_key,
                    "network_id": network_id,
                    "sample_index": sample_index,
                    "z_rate": float(value),
                }
                for sample_index, value in enumerate(vector)
            )

    return (
        pd.DataFrame(feature_rows),
        pd.DataFrame(vector_rows),
        pd.DataFrame(activity_rows),
        pd.DataFrame(predecision_rows),
    )

def load_cbgt_state_bins(network_manifest: Path, networks_dir: Path) -> pd.DataFrame:
    manifest = read_csv(network_manifest)
    tables = []
    for row in manifest.itertuples(index=False):
        network_id = int(row.network_id)
        path = Path(networks_dir) / f"network_{network_id}" / "data_conf.csv"
        df = _standardize_cbgt_rows(read_csv(path)).query("phase == 0").copy()
        df["network_id"] = network_id
        df["session"] = f"network_{network_id}"
        df["dataset"] = "CBGT"
        df["state"] = sum(
            df[name].astype(int) * (2 ** (3 - i))
            for i, name in enumerate(SPN_NAMES)
        )
        tables.append(
            df[
                [
                    "dataset", "session", "network_id", "trial_id", "phase",
                    "bin_index", "choice_left", "decision_time_ms", *SPN_NAMES, "state",
                ]
            ]
        )
    return pd.concat(tables, ignore_index=True)

def _optional_array(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def _first_optional_array(*paths: Path) -> np.ndarray | None:
    for path in paths:
        values = _optional_array(path)
        if values is not None:
            return values
    return None


def _steinmetz_choice_codes(
    session_dir: Path,
    choice: np.ndarray,
    included: np.ndarray,
) -> tuple[int, int]:
    contrast_left = _first_optional_array(
        session_dir / "trials.visualStim_contrastLeft.npy",
        session_dir / "trials.contrastLeft.npy",
    )
    contrast_right = _first_optional_array(
        session_dir / "trials.visualStim_contrastRight.npy",
        session_dir / "trials.contrastRight.npy",
    )
    feedback = _optional_array(session_dir / "trials.feedbackType.npy")
    if contrast_left is None or contrast_right is None or feedback is None:
        return -1, 1

    contrast_left = contrast_left.reshape(-1).astype(float)
    contrast_right = contrast_right.reshape(-1).astype(float)
    feedback = feedback.reshape(-1).astype(int)
    valid = (
        included
        & (feedback == 1)
        & np.isfinite(contrast_left)
        & np.isfinite(contrast_right)
        & (contrast_left != contrast_right)
        & (choice != 0)
    )
    left_trials = valid & (contrast_left > contrast_right)
    right_trials = valid & (contrast_right > contrast_left)
    if left_trials.sum() < 5 or right_trials.sum() < 5:
        return -1, 1

    left_code = int(pd.Series(choice[left_trials]).mode().iloc[0])
    right_code = int(pd.Series(choice[right_trials]).mode().iloc[0])
    return (left_code, right_code) if left_code != right_code and left_code != 0 and right_code != 0 else (-1, 1)


def _first_movement_times(session_dir: Path, stim_times: np.ndarray, move_intervals: np.ndarray) -> np.ndarray:
    first = _optional_array(session_dir / "trials.firstMovement_times.npy")
    if first is not None and len(first.reshape(-1)) == len(stim_times):
        return first.reshape(-1).astype(float)
    starts = np.asarray(move_intervals, float)[:, 0]
    return np.array([starts[starts > stim].min() for stim in stim_times], dtype=float)


def _movement_in_window(
    stim_times: np.ndarray,
    move_intervals: np.ndarray,
    window_s: tuple[float, float],
) -> np.ndarray:
    intervals = np.asarray(move_intervals, dtype=float)
    starts, ends = intervals[:, 0], intervals[:, 1]
    return np.array([
        np.any((starts < stim + window_s[1]) & (ends > stim + window_s[0]))
        for stim in np.asarray(stim_times, dtype=float)
    ])


def _cluster_regions(session_dir: Path) -> np.ndarray:
    peak = np.load(session_dir / "clusters.peakChannel.npy").reshape(-1).astype(int)
    regions = read_csv(session_dir / "channels.brainLocation.tsv", sep="\t")["allen_ontology"].astype(str).to_numpy()
    index = peak if peak.min() == 0 else peak - 1
    return regions[np.clip(index, 0, len(regions) - 1)]


def _good_steinmetz_units(session_dir: Path) -> np.ndarray:
    quality = _optional_array(session_dir / "clusters._phy_annotation.npy")
    if quality is None:
        return np.arange(len(np.load(session_dir / "clusters.peakChannel.npy").reshape(-1)), dtype=int)
    return np.flatnonzero(np.asarray(quality).reshape(-1) >= 2)


def _spike_tensor(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    unit_ids: np.ndarray,
    event_times: np.ndarray,
    edges_s: np.ndarray,
) -> np.ndarray:
    out = np.zeros((len(unit_ids), len(event_times), len(edges_s) - 1), dtype=float)
    for unit_i, unit_id in enumerate(unit_ids):
        unit_spikes = spike_times[spike_clusters == unit_id]
        for trial_i, event in enumerate(event_times):
            out[unit_i, trial_i] = np.histogram(unit_spikes - event, bins=edges_s)[0]
    return out


def load_steinmetz_session(
    session_dir: Path,
    session_name: str,
    target_region: str = "CP",
    rt_window_s: tuple[float, float] = (0.125, 0.400),
    matching_seed: int = 2,
    premove_gap_s: float = 0.010,
    spike_times_override: np.ndarray | None = None,
    spike_clusters_override: np.ndarray | None = None,
) -> dict[str, Any]:
    session_dir = Path(session_dir)
    stim = np.load(session_dir / "trials.visualStim_times.npy").reshape(-1).astype(float)
    choice = np.load(session_dir / "trials.response_choice.npy").reshape(-1).astype(int)
    included = np.load(session_dir / "trials.included.npy").reshape(-1).astype(bool)
    left_code, right_code = _steinmetz_choice_codes(session_dir, choice, included)
    moves = np.load(session_dir / "wheelMoves.intervals.npy").astype(float)
    first_move = _first_movement_times(session_dir, stim, moves)
    rt = first_move - stim
    early_move = _movement_in_window(stim, moves, (-0.050, rt_window_s[0]))
    eligible = included & np.isin(choice, [left_code, right_code]) & (rt >= rt_window_s[0]) & (rt <= rt_window_s[1]) & ~early_move
    trial_ids = np.flatnonzero(eligible)

    spike_times = (
        np.load(session_dir / "spikes.times.npy").reshape(-1).astype(float)
        if spike_times_override is None
        else np.asarray(spike_times_override, dtype=float).reshape(-1)
    )
    spike_clusters = (
        np.load(session_dir / "spikes.clusters.npy").reshape(-1).astype(int)
        if spike_clusters_override is None
        else np.asarray(spike_clusters_override, dtype=int).reshape(-1)
    )
    regions = _cluster_regions(session_dir)
    good = _good_steinmetz_units(session_dir)
    unit_ids = good[np.array([target_region in regions[i] for i in good])]

    edges = np.arange(-0.5, 0.2 + BIN_SIZE_S / 2, BIN_SIZE_S)
    centers = (edges[:-1] + edges[1:]) / 2
    counts = _spike_tensor(spike_times, spike_clusters, unit_ids, first_move[trial_ids], edges)
    rates = counts / BIN_SIZE_S
    rates = lfilter(causal_half_gaussian_kernel(), [1.0], rates, axis=-1)

    baseline_trials = np.flatnonzero(included & np.isfinite(stim))
    baseline = np.zeros(len(unit_ids), dtype=float)
    for unit_i, unit_id in enumerate(unit_ids):
        unit_spikes = spike_times[spike_clusters == unit_id]
        total = sum(
            np.sum((unit_spikes >= stim[t] + BASELINE_WINDOW_S[0]) & (unit_spikes < stim[t] + BASELINE_WINDOW_S[1]))
            for t in baseline_trials
        )
        baseline[unit_i] = total / (len(baseline_trials) * (BASELINE_WINDOW_S[1] - BASELINE_WINDOW_S[0]))
    rates /= baseline[:, None, None] + BASELINE_ADD_HZ

    contrast_left = _first_optional_array(
        session_dir / "trials.visualStim_contrastLeft.npy",
        session_dir / "trials.contrastLeft.npy",
    )
    contrast_right = _first_optional_array(
        session_dir / "trials.visualStim_contrastRight.npy",
        session_dir / "trials.contrastRight.npy",
    )
    if contrast_left is None:
        contrast_left = np.zeros_like(stim)
    if contrast_right is None:
        contrast_right = np.zeros_like(stim)
    contrast_left = np.asarray(contrast_left, dtype=float).reshape(-1)
    contrast_right = np.asarray(contrast_right, dtype=float).reshape(-1)
    signed_contrast = np.nan_to_num(contrast_right - contrast_left)[trial_ids]
    evidence_magnitude = np.abs(signed_contrast)
    choice_left = choice[trial_ids] == left_code
    left, right = match_evidence_trials(
        choice_left,
        evidence_magnitude,
        seed_text=session_name,
        seed=matching_seed,
    )
    selected = np.sort(np.concatenate([left, right]))

    return {
        "session": session_name,
        "session_dir": str(session_dir),
        "choice_left_code": left_code,
        "choice_right_code": right_code,
        "unit_ids": unit_ids,
        "trial_ids": trial_ids[selected],
        "choice_left": choice_left[selected],
        "decision_time_ms": 1000 * rt[trial_ids][selected],
        "signed_contrast": signed_contrast[selected],
        "evidence_magnitude": evidence_magnitude[selected],
        "bin_centers_s": centers,
        "rates": rates[:, selected],
        "matching_seed": int(matching_seed),
        "premove_gap_s": float(premove_gap_s),
    }


def _make_intervals(event_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    starts = (event_times[:, None] + edges[:-1]).reshape(-1)
    ends = (event_times[:, None] + edges[1:]).reshape(-1)
    return np.column_stack([starts, ends])


def _reorder_counts(
    counts: np.ndarray,
    cluster_ids: np.ndarray,
    unit_ids: np.ndarray,
) -> np.ndarray:
    out = np.zeros((len(unit_ids), counts.shape[1]), dtype=counts.dtype)
    row_by_id = {int(cluster_id): i for i, cluster_id in enumerate(cluster_ids)}
    for row, unit_id in enumerate(unit_ids):
        source = row_by_id.get(int(unit_id))
        if source is not None:
            out[row] = counts[source]
    return out


def parse_seed_list(value: Any) -> list[int]:
    """Parse one seed or a semicolon-separated sequence of fallback seeds."""
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(item.strip()) for item in str(value).split(";") if item.strip()]


def _ibl_session_metadata(one, eid: str) -> dict[str, str]:
    details = one.alyx.rest("sessions", "read", id=str(eid))
    return {
        "lab": str(details["lab"]),
        "subject": str(details["subject"]),
        "date": str(details["start_time"])[:10],
    }


def resolve_ibl_candidates(
    one,
    lab: str,
    subject: str,
    date: str,
    max_insertions_to_process: int,
    random_seeds: Any,
    all_pids: np.ndarray | list | None = None,
    metadata_cache: dict[str, dict[str, str]] | None = None,
    target_region: str = "CP",
    project: str = "brainwide",
) -> pd.DataFrame:
    """Recover the saved IBL insertion scan for one manuscript session."""
    if all_pids is None:
        all_pids = one.search_insertions(
            atlas_acronym=target_region,
            datasets="spikes.times.npy",
            project=project,
        )
    if metadata_cache is None:
        metadata_cache = {}

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for seed in parse_seed_list(random_seeds):
        ordered = np.asarray(list(all_pids), dtype=object).copy()
        np.random.default_rng(seed).shuffle(ordered)
        for scan_rank, pid in enumerate(ordered[: int(max_insertions_to_process)], start=1):
            pid = str(pid)
            eid, probe = one.pid2eid(pid)
            eid = str(eid)
            if eid not in metadata_cache:
                metadata_cache[eid] = _ibl_session_metadata(one, eid)
            metadata = metadata_cache[eid]
            matches = (
                metadata["lab"].lower() == str(lab).lower()
                and metadata["subject"] == str(subject)
                and metadata["date"] == str(date)
            )
            if matches and pid not in seen:
                rows.append(
                    {
                        "pid": pid,
                        "eid": eid,
                        "probe": str(probe),
                        "resolution_seed": int(seed),
                        "scan_rank": int(scan_rank),
                    }
                )
                seen.add(pid)

    return pd.DataFrame(rows)


def load_ibl_insertion(
    one,
    pid: str,
    session_name: str,
    rt_window_s: tuple[float, float] = (0.080, 4.000),
    matching_minimum_per_bin: int = 10,
    matching_seed: int = 20,
    matching_key_suffix: str = "",
    premove_gap_s: float = 0.010,
    minimum_trials: int = 0,
    minimum_trials_each_side: int = 0,
    minimum_good_units: int = 1,
    target_region: str = "CP",
    require_resolved_histology: bool = True,
) -> dict[str, Any]:
    """Load and preprocess one public IBL Brain-Wide Map insertion."""
    from brainbox.io.one import SpikeSortingLoader
    from brainbox.population.decode import get_spike_counts_in_bins

    eid, probe = one.pid2eid(pid)
    trials = one.load_object(eid, "trials", collection="alf")
    required = [
        "choice",
        "probabilityLeft",
        "feedbackType",
        "feedback_times",
        "stimOn_times",
        "firstMovement_times",
    ]
    trial_arrays = {name: np.asarray(trials[name]).reshape(-1) for name in required}
    choice = trial_arrays["choice"]
    stim = trial_arrays["stimOn_times"].astype(float)
    first_move = trial_arrays["firstMovement_times"].astype(float)
    rt = first_move - stim

    mask = np.logical_and.reduce(
        [np.isfinite(values.astype(float)) for values in trial_arrays.values()]
    )
    mask &= np.isin(choice, [-1, 1])
    mask &= (rt >= rt_window_s[0]) & (rt <= rt_window_s[1])
    eligible_trial_ids = np.flatnonzero(mask)
    if len(eligible_trial_ids) < int(minimum_trials):
        raise ValueError(
            f"{session_name}: {len(eligible_trial_ids)} eligible trials; "
            f"at least {minimum_trials} are required."
        )

    c_left = np.nan_to_num(
        np.asarray(trials.get("contrastLeft", np.zeros(len(choice))), float)
    )[eligible_trial_ids]
    c_right = np.nan_to_num(
        np.asarray(trials.get("contrastRight", np.zeros(len(choice))), float)
    )[eligible_trial_ids]
    signed_contrast = c_right - c_left
    choice_left = choice[eligible_trial_ids] == -1
    left, right = match_choice_trials(
        choice_left,
        signed_contrast,
        minimum_per_bin=matching_minimum_per_bin,
        seed_text=f"{eid}_{probe}{matching_key_suffix}",
        seed=matching_seed,
    )
    if min(len(left), len(right)) < int(minimum_trials_each_side):
        raise ValueError(
            f"{session_name}: too few matched trials per choice "
            f"(left={len(left)}, right={len(right)})."
        )
    selected = np.sort(np.concatenate([left, right]))
    trial_ids = eligible_trial_ids[selected]

    loader = SpikeSortingLoader(pid=pid, one=one)
    loader.load_channels()
    if require_resolved_histology and loader.histology not in ("resolved", "alf"):
        raise ValueError(f"{session_name}: unresolved histology ({loader.histology!r}).")
    spikes, clusters, channels = loader.load_spike_sorting()
    clusters = loader.merge_clusters(spikes, clusters, channels)

    acronyms = np.array(
        [x.decode() if isinstance(x, bytes) else str(x) for x in clusters["acronym"]]
    )
    in_region = (
        np.isin(acronyms, [target_region, "STR", "STRd"])
        | np.char.startswith(acronyms.astype(str), target_region)
    )
    unit_ids = np.asarray(clusters["cluster_id"])[
        (np.asarray(clusters["label"]) == 1) & in_region
    ].astype(int)
    if len(unit_ids) < int(minimum_good_units):
        raise ValueError(
            f"{session_name}: {len(unit_ids)} good {target_region} units; "
            f"at least {minimum_good_units} are required."
        )

    spike_times = np.asarray(spikes["times"], float)
    spike_clusters = np.asarray(spikes["clusters"], int)
    keep = np.isin(spike_clusters, unit_ids)
    spike_times = spike_times[keep]
    spike_clusters = spike_clusters[keep]

    baseline_intervals = np.column_stack(
        [
            stim[eligible_trial_ids] + BASELINE_WINDOW_S[0],
            stim[eligible_trial_ids] + BASELINE_WINDOW_S[1],
        ]
    )
    baseline_counts, baseline_ids = get_spike_counts_in_bins(
        spike_times, spike_clusters, baseline_intervals
    )
    baseline_counts = _reorder_counts(baseline_counts, baseline_ids, unit_ids)
    baseline = baseline_counts.mean(axis=1) / (
        BASELINE_WINDOW_S[1] - BASELINE_WINDOW_S[0]
    )

    edges = np.arange(-0.7, 0.2 + BIN_SIZE_S / 2, BIN_SIZE_S)
    centers = (edges[:-1] + edges[1:]) / 2
    intervals = _make_intervals(first_move[trial_ids], edges)
    counts, count_ids = get_spike_counts_in_bins(
        spike_times, spike_clusters, intervals
    )
    counts = _reorder_counts(counts, count_ids, unit_ids)
    rates = counts.reshape(len(unit_ids), len(trial_ids), len(centers)) / BIN_SIZE_S

    rates /= baseline[:, None, None] + BASELINE_ADD_HZ

    return {
        "session": session_name,
        "eid": str(eid),
        "pid": str(pid),
        "probe": str(probe),
        "unit_ids": unit_ids,
        "trial_ids": trial_ids.astype(int),
        "choice_left": choice[trial_ids] == -1,
        "decision_time_ms": 1000 * rt[trial_ids],
        "signed_contrast": signed_contrast[selected],
        "bin_centers_s": centers,
        "rates": rates,
        "premove_gap_s": float(premove_gap_s),
        "matching_seed": int(matching_seed),
        "min_profile_range": 0.0,
        "ramp_min_slope": 1e-4,
        "minimum_good_units": int(minimum_good_units),
        "minimum_subcluster_size": 3,
    }


_EXACT_EPS = 1e-8


def _exact_session_rng(text: str, seed: int = 0) -> np.random.Generator:
    digest = hashlib.md5(str(text).encode("utf-8")).digest()
    local_seed = (int.from_bytes(digest[:4], "little") + int(seed)) % 2**32
    return np.random.default_rng(local_seed)


def _exact_optional_array(path: Path) -> np.ndarray | None:
    path = Path(path)
    return np.load(path) if path.exists() else None


def _exact_ravel(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values).reshape(-1))


def _steinmetz_choice_codes_exact(
    session_dir: Path,
    choice: np.ndarray,
    included: np.ndarray,
) -> tuple[int, int]:
    """Infer the left/right response codes exactly as in the source notebook."""
    choice = _exact_ravel(choice).astype(int)
    included = _exact_ravel(included).astype(bool)

    contrast_left = _exact_optional_array(session_dir / "trials.visualStim_contrastLeft.npy")
    if contrast_left is None:
        contrast_left = _exact_optional_array(session_dir / "trials.contrastLeft.npy")
    contrast_right = _exact_optional_array(session_dir / "trials.visualStim_contrastRight.npy")
    if contrast_right is None:
        contrast_right = _exact_optional_array(session_dir / "trials.contrastRight.npy")
    feedback = _exact_optional_array(session_dir / "trials.feedbackType.npy")

    if contrast_left is None or contrast_right is None or feedback is None:
        return -1, 1

    contrast_left = _exact_ravel(contrast_left).astype(float)
    contrast_right = _exact_ravel(contrast_right).astype(float)
    feedback = _exact_ravel(feedback).astype(int)

    valid = (
        included
        & (feedback == 1)
        & np.isfinite(contrast_left)
        & np.isfinite(contrast_right)
        & (contrast_left != contrast_right)
        & (choice != 0)
    )
    if valid.sum() < 20:
        return -1, 1

    left_should = valid & (contrast_left > contrast_right)
    right_should = valid & (contrast_right > contrast_left)
    if left_should.sum() < 5 or right_should.sum() < 5:
        return -1, 1

    left_code = int(pd.Series(choice[left_should]).mode().iloc[0])
    right_code = int(pd.Series(choice[right_should]).mode().iloc[0])
    if left_code == 0 or right_code == 0 or left_code == right_code:
        return -1, 1
    return left_code, right_code


def _steinmetz_movement_onsets_exact(
    session_dir: Path,
    stimulus_times: np.ndarray,
    movement_intervals: np.ndarray,
) -> np.ndarray:
    stimulus_times = _exact_ravel(stimulus_times).astype(float)
    first_movement = _exact_optional_array(session_dir / "trials.firstMovement_times.npy")
    if first_movement is not None:
        first_movement = _exact_ravel(first_movement).astype(float)
        if len(first_movement) == len(stimulus_times):
            return first_movement

    starts = np.asarray(movement_intervals, dtype=float)[:, 0]
    onsets = np.full(len(stimulus_times), np.nan, dtype=float)
    for trial, stimulus_time in enumerate(stimulus_times):
        after = starts[starts > stimulus_time]
        if after.size:
            onsets[trial] = after.min()
    return onsets


def _steinmetz_movement_overlap_exact(
    movement_starts: np.ndarray,
    movement_ends: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
) -> np.ndarray:
    overlap = np.zeros(len(window_starts), dtype=bool)
    for trial, (start, end) in enumerate(zip(window_starts, window_ends)):
        if not (np.isfinite(start) and np.isfinite(end)):
            overlap[trial] = True
            continue
        overlap[trial] = np.any((movement_starts < end) & (movement_ends > start))
    return overlap


def _steinmetz_match_trials_exact(
    session_dir: Path,
    trials_left: np.ndarray,
    trials_right: np.ndarray,
    included: np.ndarray,
    matching_seed: int,
    n_bins: int = 3,
    minimum_per_bin: int = 5,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Match choices by absolute evidence using the exact source algorithm."""
    contrast_left = _exact_optional_array(session_dir / "trials.visualStim_contrastLeft.npy")
    if contrast_left is None:
        contrast_left = _exact_optional_array(session_dir / "trials.contrastLeft.npy")
    contrast_right = _exact_optional_array(session_dir / "trials.visualStim_contrastRight.npy")
    if contrast_right is None:
        contrast_right = _exact_optional_array(session_dir / "trials.contrastRight.npy")

    if contrast_left is None or contrast_right is None:
        return trials_left, trials_right, False

    contrast_left = _exact_ravel(contrast_left).astype(float)
    contrast_right = _exact_ravel(contrast_right).astype(float)
    included = _exact_ravel(included).astype(bool)
    evidence = np.abs(contrast_right - contrast_left)
    valid = included & np.isfinite(evidence)

    left = np.asarray(trials_left, dtype=int)[valid[np.asarray(trials_left, dtype=int)]]
    right = np.asarray(trials_right, dtype=int)[valid[np.asarray(trials_right, dtype=int)]]
    if left.size == 0 or right.size == 0:
        return trials_left, trials_right, False

    evidence_values = np.unique(evidence[valid])
    if evidence_values.size < n_bins:
        return trials_left, trials_right, False

    groups = np.array_split(np.sort(evidence_values), n_bins)
    rng = _exact_session_rng(Path(session_dir).name, matching_seed)
    left_out: list[np.ndarray] = []
    right_out: list[np.ndarray] = []
    for group in groups:
        low, high = float(group.min()), float(group.max())
        left_bin = left[(evidence[left] >= low) & (evidence[left] <= high)]
        right_bin = right[(evidence[right] >= low) & (evidence[right] <= high)]
        n_take = min(left_bin.size, right_bin.size)
        if n_take < minimum_per_bin:
            continue
        left_out.append(rng.choice(left_bin, size=n_take, replace=False))
        right_out.append(rng.choice(right_bin, size=n_take, replace=False))

    if not left_out:
        return trials_left, trials_right, False
    return (
        np.sort(np.concatenate(left_out)),
        np.sort(np.concatenate(right_out)),
        True,
    )


def _steinmetz_cluster_regions_exact(session_dir: Path) -> np.ndarray:
    peak_channel = np.load(session_dir / "clusters.peakChannel.npy").astype(int)
    regions = read_csv(session_dir / "channels.brainLocation.tsv", sep="\t")[
        "allen_ontology"
    ].astype(str).to_numpy()
    index = peak_channel if peak_channel.min() == 0 else peak_channel - 1
    return regions[np.clip(index, 0, len(regions) - 1)]


def _steinmetz_good_cluster_ids_exact(session_dir: Path) -> np.ndarray:
    annotation_path = session_dir / "clusters._phy_annotation.npy"
    if not annotation_path.exists():
        n_clusters = len(np.load(session_dir / "clusters.peakChannel.npy"))
        return np.arange(n_clusters, dtype=int)
    annotations = np.load(annotation_path)
    return np.where(annotations >= 2)[0].astype(int)


def load_steinmetz_session_exact(
    session_dir: Path,
    session_name: str,
    matching_seed: int,
    premove_gap_s: float = 0.010,
    target_region: str = "CP",
) -> dict[str, Any]:
    """Load one Steinmetz session without altering the original pipeline."""
    session_dir = Path(session_dir)
    spike_times = np.load(session_dir / "spikes.times.npy").astype(float)
    spike_clusters = np.load(session_dir / "spikes.clusters.npy").astype(int)

    cluster_regions = _steinmetz_cluster_regions_exact(session_dir)
    good_cluster_ids = _steinmetz_good_cluster_ids_exact(session_dir)
    target_cluster_ids = np.where(cluster_regions == target_region)[0].astype(int)
    unit_ids_all = np.intersect1d(good_cluster_ids, target_cluster_ids)
    if unit_ids_all.size < 8:
        raise ValueError(
            f"{session_name}: only {unit_ids_all.size} good {target_region} units before filtering."
        )

    stimulus_times = _exact_ravel(
        np.load(session_dir / "trials.visualStim_times.npy")
    ).astype(float)
    choice = _exact_ravel(
        np.load(session_dir / "trials.response_choice.npy")
    ).astype(int)
    included = _exact_ravel(
        np.load(session_dir / "trials.included.npy")
    ).astype(bool)
    movement_intervals = np.load(session_dir / "wheelMoves.intervals.npy").astype(float)
    movement_starts = movement_intervals[:, 0].astype(float)
    movement_ends = movement_intervals[:, 1].astype(float)

    choice_left_code, choice_right_code = _steinmetz_choice_codes_exact(
        session_dir, choice, included
    )
    movement_onsets = _steinmetz_movement_onsets_exact(
        session_dir, stimulus_times, movement_intervals
    )
    reaction_time_s = movement_onsets - stimulus_times
    in_rt = (
        np.isfinite(reaction_time_s)
        & (reaction_time_s >= 0.125)
        & (reaction_time_s <= 0.400)
    )
    early_movement = _steinmetz_movement_overlap_exact(
        movement_starts,
        movement_ends,
        stimulus_times - 0.05,
        stimulus_times + 0.125,
    )
    usable = included & in_rt & (~early_movement) & (choice != 0)
    trials_left = np.where(usable & (choice == choice_left_code))[0]
    trials_right = np.where(usable & (choice == choice_right_code))[0]
    if trials_left.size < 12 or trials_right.size < 12:
        raise ValueError(
            f"{session_name}: insufficient original eligible trials "
            f"(left={trials_left.size}, right={trials_right.size})."
        )

    trials_left, trials_right, contrast_matched = _steinmetz_match_trials_exact(
        session_dir,
        trials_left,
        trials_right,
        included,
        matching_seed=matching_seed,
        n_bins=3,
        minimum_per_bin=5,
    )

    contrast_left = _exact_optional_array(session_dir / "trials.visualStim_contrastLeft.npy")
    if contrast_left is None:
        contrast_left = _exact_optional_array(session_dir / "trials.contrastLeft.npy")
    contrast_right = _exact_optional_array(session_dir / "trials.visualStim_contrastRight.npy")
    if contrast_right is None:
        contrast_right = _exact_optional_array(session_dir / "trials.contrastRight.npy")
    if contrast_left is None or contrast_right is None:
        signed_contrast = np.full(len(choice), np.nan, dtype=float)
    else:
        signed_contrast = _exact_ravel(contrast_right).astype(float) - _exact_ravel(
            contrast_left
        ).astype(float)

    return {
        "dataset": "Steinmetz",
        "session": str(session_name),
        "session_folder": session_dir.name,
        "session_dir": str(session_dir),
        "matching_seed": int(matching_seed),
        "premove_gap_s": float(premove_gap_s),
        "spike_times": spike_times,
        "spike_clusters": spike_clusters,
        "unit_ids_all": unit_ids_all.astype(int),
        "stimulus_times": _exact_ravel(stimulus_times).astype(float),
        "choice": _exact_ravel(choice).astype(int),
        "included": _exact_ravel(included).astype(bool),
        "movement_intervals": movement_intervals,
        "movement_onsets": _exact_ravel(movement_onsets).astype(float),
        "reaction_time_s": _exact_ravel(reaction_time_s).astype(float),
        "usable": _exact_ravel(usable).astype(bool),
        "trials_left": np.asarray(trials_left, dtype=int),
        "trials_right": np.asarray(trials_right, dtype=int),
        "baseline_trials": _exact_ravel(included).astype(bool).copy(),
        "choice_left_code": int(choice_left_code),
        "choice_right_code": int(choice_right_code),
        "contrast_matched": bool(contrast_matched),
        "signed_contrast_all": signed_contrast,
        "evidence_magnitude_all": np.abs(signed_contrast),
    }


def _ibl_time_edges_exact(
    t_pre_s: float = 0.70,
    t_post_s: float = 0.20,
    bin_size_s: float = 0.010,
) -> np.ndarray:
    n_bins = int(round((t_pre_s + t_post_s) / bin_size_s))
    edges = -t_pre_s + np.arange(n_bins + 1) * bin_size_s
    edges[0] = -t_pre_s
    edges[-1] = t_post_s
    return edges


def _ibl_trial_mask_exact(
    trials: Any,
    rt_min_s: float = 0.080,
    rt_max_s: float = 4.000,
) -> np.ndarray:
    required = (
        "choice",
        "probabilityLeft",
        "feedbackType",
        "feedback_times",
        "stimOn_times",
        "firstMovement_times",
    )
    for field in required:
        if field not in trials:
            raise KeyError(f"Missing trials field {field!r}.")

    arrays = {field: np.asarray(trials[field]).reshape(-1) for field in required}
    finite = np.logical_and.reduce(
        [np.isfinite(values.astype(float)) for values in arrays.values()]
    )
    reaction_time = (
        arrays["firstMovement_times"].astype(float)
        - arrays["stimOn_times"].astype(float)
    )
    return finite & np.isfinite(reaction_time) & (reaction_time >= rt_min_s) & (
        reaction_time <= rt_max_s
    )


def _ibl_match_trials_by_signed_contrast_exact(
    choice: np.ndarray,
    signed_contrast: np.ndarray,
    n_bins: int,
    minimum_per_bin: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    choice = np.asarray(choice).reshape(-1)
    contrast = np.asarray(signed_contrast).reshape(-1).copy()
    left = np.where(choice == -1)[0]
    right = np.where(choice == 1)[0]

    contrast[~np.isfinite(contrast)] = 0.0
    edges = np.unique(np.quantile(contrast, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return left, right

    left_keep: list[np.ndarray] = []
    right_keep: list[np.ndarray] = []
    for edge_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if high == edges[-1]:
            left_bin = left[(contrast[left] >= low) & (contrast[left] <= high)]
            right_bin = right[(contrast[right] >= low) & (contrast[right] <= high)]
        else:
            left_bin = left[(contrast[left] >= low) & (contrast[left] < high)]
            right_bin = right[(contrast[right] >= low) & (contrast[right] < high)]
        n_take = min(len(left_bin), len(right_bin))
        if n_take < minimum_per_bin:
            continue
        left_keep.append(rng.choice(left_bin, size=n_take, replace=False))
        right_keep.append(rng.choice(right_bin, size=n_take, replace=False))

    if not left_keep:
        return left, right
    return np.sort(np.concatenate(left_keep)), np.sort(np.concatenate(right_keep))


def _ibl_session_metadata_exact(one: Any, pid: str) -> dict[str, Any]:
    eid, probe = one.pid2eid(str(pid))
    session_path = Path(one.eid2path(eid))
    parts = session_path.parts
    if "Subjects" in parts:
        index = parts.index("Subjects")
        lab = parts[index - 1]
        subject = parts[index + 1]
        date = parts[index + 2]
    else:
        lab, subject, date = "", "", ""
    return {
        "pid": str(pid),
        "eid": str(eid),
        "probe": str(probe),
        "session_path": str(session_path),
        "lab": str(lab),
        "subject": str(subject),
        "date": str(date),
    }


def _parse_seed_sequence(value: Any) -> list[int]:
    if isinstance(value, str):
        tokens = [token.strip() for token in value.replace(",", ";").split(";")]
        return [int(token) for token in tokens if token]
    if np.isscalar(value):
        return [int(value)]
    return [int(seed) for seed in value]


def resolve_ibl_candidates_exact(
    one: Any,
    lab: str,
    subject: str,
    date: str,
    max_insertions_to_process: int,
    random_seeds: Any,
    all_pids: list[str] | np.ndarray | None = None,
    metadata_cache: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if all_pids is None:
        all_pids = one.search_insertions(
            atlas_acronym="CP",
            datasets="spikes.times.npy",
            project="brainwide",
        )
    all_pids = [str(pid) for pid in all_pids]
    metadata_cache = {} if metadata_cache is None else metadata_cache

    rows: list[dict[str, Any]] = []
    for seed in _parse_seed_sequence(random_seeds):
        shuffled = list(all_pids)
        np.random.default_rng(seed).shuffle(shuffled)
        for scan_rank, pid in enumerate(
            shuffled[: int(max_insertions_to_process)], start=1
        ):
            try:
                metadata = metadata_cache.get(pid)
                if metadata is None:
                    metadata = _ibl_session_metadata_exact(one, pid)
                    metadata_cache[pid] = metadata
            except Exception:
                continue
            if (
                metadata["lab"].lower() == str(lab).lower()
                and metadata["subject"] == str(subject)
                and metadata["date"] == str(date)
            ):
                rows.append(
                    {
                        **metadata,
                        "resolution_seed": int(seed),
                        "scan_rank": int(scan_rank),
                        "max_insertions_to_process": int(max_insertions_to_process),
                    }
                )
    return pd.DataFrame(rows)


def load_ibl_insertion_exact(
    one: Any,
    pid: str,
    session_name: str,
    random_seed: int,
    rt_window_s: tuple[float, float] = (0.080, 4.000),
    premove_gap_s: float = 0.010,
    matching_minimum_per_bin: int = 10,
    minimum_trials: int = 400,
    minimum_trials_each_side: int = 50,
    minimum_good_units: int = 8,
    target_region: str = "CP",
) -> dict[str, Any]:
    from brainbox.io.one import SpikeSortingLoader
    from brainbox.population.decode import get_spike_counts_in_bins

    eid, probe = one.pid2eid(str(pid))
    session_path = str(one.eid2path(eid))
    trials = one.load_object(eid, "trials", collection="alf")
    n_trials_total = len(trials["choice"])

    mask = _ibl_trial_mask_exact(trials, *rt_window_s)
    choice_all = np.asarray(trials["choice"]).reshape(-1)
    mask &= np.isin(choice_all, [-1, 1])
    trial_ids = np.where(mask)[0]
    if trial_ids.size < minimum_trials:
        raise ValueError(
            f"{session_name}: {trial_ids.size} eligible trials; at least {minimum_trials} required."
        )

    stimulus = np.asarray(trials["stimOn_times"], dtype=float)[trial_ids]
    first_movement = np.asarray(trials["firstMovement_times"], dtype=float)[trial_ids]
    reaction_time = first_movement - stimulus
    choice = choice_all[trial_ids].astype(int)

    contrast_left_all = np.asarray(
        trials.get("contrastLeft", np.zeros(n_trials_total)), dtype=float
    )
    contrast_right_all = np.asarray(
        trials.get("contrastRight", np.zeros(n_trials_total)), dtype=float
    )
    contrast_left = np.nan_to_num(contrast_left_all[trial_ids], nan=0.0)
    contrast_right = np.nan_to_num(contrast_right_all[trial_ids], nan=0.0)
    signed_contrast = contrast_right - contrast_left

    loader = SpikeSortingLoader(pid=str(pid), one=one)
    loader.load_channels()
    if loader.histology not in ("resolved", "alf"):
        raise ValueError(f"{session_name}: histology={loader.histology!r}.")
    spikes, clusters, channels = loader.load_spike_sorting()
    clusters = loader.merge_clusters(spikes, clusters, channels)

    acronyms = np.array(
        [
            acronym.decode() if isinstance(acronym, (bytes, np.bytes_)) else str(acronym)
            for acronym in clusters["acronym"]
        ],
        dtype=object,
    )
    in_region = np.isin(acronyms, [target_region, "STR", "STRd"]) | np.array(
        [str(acronym).startswith(target_region) for acronym in acronyms], dtype=bool
    )
    unit_ids = np.asarray(clusters["cluster_id"])[
        (np.asarray(clusters["label"]) == 1) & in_region
    ].astype(int)
    if unit_ids.size < minimum_good_units:
        raise ValueError(
            f"{session_name}: {unit_ids.size} good {target_region} units; "
            f"at least {minimum_good_units} required."
        )

    spike_times = np.asarray(spikes["times"], dtype=float)
    spike_clusters = np.asarray(spikes["clusters"], dtype=int)
    keep_spikes = np.isin(spike_clusters, unit_ids)
    spike_times = spike_times[keep_spikes]
    spike_clusters = spike_clusters[keep_spikes]

    baseline_intervals = np.column_stack(
        [stimulus - 0.20, stimulus]
    )
    baseline_counts, baseline_ids = get_spike_counts_in_bins(
        spike_times, spike_clusters, baseline_intervals
    )
    baseline_counts = _reorder_counts(baseline_counts, baseline_ids, unit_ids)
    baseline_rate = baseline_counts.mean(axis=1) / 0.20

    edges = _ibl_time_edges_exact()
    centers = (edges[:-1] + edges[1:]) / 2
    intervals = _make_intervals(first_movement, edges)
    counts, count_ids = get_spike_counts_in_bins(
        spike_times, spike_clusters, intervals
    )
    counts = _reorder_counts(counts, count_ids, unit_ids)
    rates = counts.reshape(unit_ids.size, trial_ids.size, len(centers)) / 0.010
    rates = rates / (baseline_rate[:, None, None] + 1.0)

    rng = _exact_session_rng(f"{eid}_{probe}", random_seed)
    idx_left, idx_right = _ibl_match_trials_by_signed_contrast_exact(
        choice,
        signed_contrast,
        n_bins=3,
        minimum_per_bin=matching_minimum_per_bin,
        rng=rng,
    )
    if min(idx_left.size, idx_right.size) < minimum_trials_each_side:
        raise ValueError(
            f"{session_name}: too few matched trials "
            f"(left={idx_left.size}, right={idx_right.size})."
        )

    return {
        "dataset": "IBL",
        "session": str(session_name),
        "session_path": session_path,
        "eid": str(eid),
        "pid": str(pid),
        "probe": str(probe),
        "random_seed": int(random_seed),
        "premove_gap_s": float(premove_gap_s),
        "n_trials_total": int(n_trials_total),
        "trial_ids": trial_ids.astype(int),
        "stimulus_times": stimulus.astype(float),
        "first_movement_times": first_movement.astype(float),
        "reaction_time_s": reaction_time.astype(float),
        "decision_time_ms": (1000.0 * reaction_time).astype(float),
        "choice": choice.astype(int),
        "choice_left": (choice == -1),
        "signed_contrast": signed_contrast.astype(float),
        "idxL": idx_left.astype(int),
        "idxR": idx_right.astype(int),
        "selected_local_trials": np.sort(
            np.concatenate([idx_left, idx_right])
        ).astype(int),
        "unit_ids": unit_ids.astype(int),
        "bin_edges_s": edges.astype(float),
        "bin_centers_s": centers.astype(float),
        "rates": rates.astype(float),
        "baseline_rate_hz": baseline_rate.astype(float),
    }


def load_steinmetz_session_exact(
    session_dir: Path,
    session_name: str,
    matching_seed: int,
    premove_gap_s: float = 0.010,
    target_region: str = "CP",
) -> dict[str, Any]:
    """Load one Steinmetz session exactly as used for the manuscript labels."""
    session_dir = Path(session_dir)

    spike_times = np.load(session_dir / "spikes.times.npy").astype(float)
    spike_clusters = np.load(session_dir / "spikes.clusters.npy").astype(int)

    cluster_regions = _steinmetz_cluster_regions_exact(session_dir)
    good_cluster_ids = _steinmetz_good_cluster_ids_exact(session_dir)
    target_cluster_ids = np.where(cluster_regions == target_region)[0].astype(int)
    unit_ids_all = np.intersect1d(good_cluster_ids, target_cluster_ids)
    if unit_ids_all.size < 8:
        raise ValueError(
            f"{session_name}: only {unit_ids_all.size} good {target_region} units before filtering."
        )

    stimulus_times_raw = np.load(
        session_dir / "trials.visualStim_times.npy"
    ).astype(float)
    choice_raw = np.load(
        session_dir / "trials.response_choice.npy"
    ).astype(int)
    included_raw = np.load(
        session_dir / "trials.included.npy"
    ).astype(bool)

    stimulus_times = _exact_ravel(stimulus_times_raw).astype(float)
    choice = _exact_ravel(choice_raw).astype(int)
    included = _exact_ravel(included_raw).astype(bool)

    movement_intervals = np.load(
        session_dir / "wheelMoves.intervals.npy"
    ).astype(float)
    movement_starts = movement_intervals[:, 0].astype(float)
    movement_ends = movement_intervals[:, 1].astype(float)

    choice_left_code, choice_right_code = _steinmetz_choice_codes_exact(
        session_dir, choice_raw, included_raw
    )
    movement_onsets = _steinmetz_movement_onsets_exact(
        session_dir, stimulus_times_raw, movement_intervals
    )

    reaction_time_raw = movement_onsets - stimulus_times_raw
    in_rt_raw = (
        np.isfinite(reaction_time_raw)
        & (reaction_time_raw >= 0.125)
        & (reaction_time_raw <= 0.400)
    )
    early_movement_raw = _steinmetz_movement_overlap_exact(
        movement_starts,
        movement_ends,
        stimulus_times_raw - 0.05,
        stimulus_times_raw + 0.125,
    )
    usable_raw = (
        included_raw
        & in_rt_raw
        & (~early_movement_raw)
        & (choice_raw != 0)
    )
    trials_left = np.where(
        usable_raw & (choice_raw == choice_left_code)
    )[0]
    trials_right = np.where(
        usable_raw & (choice_raw == choice_right_code)
    )[0]
    if trials_left.size < 12 or trials_right.size < 12:
        raise ValueError(
            f"{session_name}: insufficient original eligible trials "
            f"(left={trials_left.size}, right={trials_right.size})."
        )

    trials_left, trials_right, contrast_matched = _steinmetz_match_trials_exact(
        session_dir,
        trials_left,
        trials_right,
        included_raw,
        matching_seed=matching_seed,
        n_bins=3,
        minimum_per_bin=5,
    )

    reaction_time_s = movement_onsets - stimulus_times
    in_rt = (
        np.isfinite(reaction_time_s)
        & (reaction_time_s >= 0.125)
        & (reaction_time_s <= 0.400)
    )
    early_movement = _steinmetz_movement_overlap_exact(
        movement_starts,
        movement_ends,
        stimulus_times - 0.05,
        stimulus_times + 0.125,
    )
    usable = included & in_rt & (~early_movement) & (choice != 0)

    contrast_left = _exact_optional_array(
        session_dir / "trials.visualStim_contrastLeft.npy"
    )
    if contrast_left is None:
        contrast_left = _exact_optional_array(
            session_dir / "trials.contrastLeft.npy"
        )
    contrast_right = _exact_optional_array(
        session_dir / "trials.visualStim_contrastRight.npy"
    )
    if contrast_right is None:
        contrast_right = _exact_optional_array(
            session_dir / "trials.contrastRight.npy"
        )
    if contrast_left is None or contrast_right is None:
        signed_contrast = np.full(len(choice), np.nan, dtype=float)
    else:
        signed_contrast = (
            _exact_ravel(contrast_right).astype(float)
            - _exact_ravel(contrast_left).astype(float)
        )

    return {
        "dataset": "Steinmetz",
        "session": str(session_name),
        "session_folder": session_dir.name,
        "session_dir": str(session_dir),
        "matching_seed": int(matching_seed),
        "premove_gap_s": float(premove_gap_s),
        "spike_times": spike_times,
        "spike_clusters": spike_clusters,
        "unit_ids_all": unit_ids_all.astype(int),
        "stimulus_times": stimulus_times,
        "choice": choice,
        "included": included,
        "movement_intervals": movement_intervals,
        "movement_onsets": _exact_ravel(movement_onsets).astype(float),
        "reaction_time_s": _exact_ravel(reaction_time_s).astype(float),
        "usable": _exact_ravel(usable).astype(bool),
        "baseline_trials": included.copy(),
        "stimulus_times_raw": stimulus_times_raw,
        "choice_raw": choice_raw,
        "included_raw": included_raw,
        "reaction_time_raw": reaction_time_raw,
        "usable_raw": usable_raw,
        "trials_left": np.asarray(trials_left, dtype=int),
        "trials_right": np.asarray(trials_right, dtype=int),
        "choice_left_code": int(choice_left_code),
        "choice_right_code": int(choice_right_code),
        "contrast_matched": bool(contrast_matched),
        "signed_contrast_all": signed_contrast,
        "evidence_magnitude_all": np.abs(signed_contrast),
    }
