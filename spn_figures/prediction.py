from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


N_BOOTSTRAP = 5000
PREDICTION_RANDOM_SEED = 20260529
UNCERTAINTY_BAND = (0.45, 0.55)
STATE_IDS = tuple(range(16))
TRANSITION_PAIRS = ((8, 10), (4, 5), (10, 11), (5, 7), (12, 13), (12, 14))

PANEL_A_BOOTSTRAP_COLUMNS = {
    "i_only": "p1_GU_visited_1_2_3_P_left",
    "left": "p1_GL_visited_8_or_10_P_left",
    "right": "p1_GR_visited_4_or_5_P_left",
}
PANEL_B_BOOTSTRAP_COLUMNS = {
    "8": "p2_left_channel_8_to_10_P_END_after_state_8",
    "8_to_10": "p2_left_channel_8_to_10_P_END_after_state_10_given_direct_8_to_10",
    "4": "p2_right_channel_4_to_5_P_END_after_state_4",
    "4_to_5": "p2_right_channel_4_to_5_P_END_after_state_5_given_direct_4_to_5",
}
PANEL_PAIRWISE_COMPARISONS = {
    "A": (("i_only", "left"), ("i_only", "right")),
    "B": (("8", "8_to_10"), ("4", "4_to_5")),
    "C": (("state10", "10_to_11"), ("state5", "5_to_7")),
}
PLOT_KEY_LABELS = {
    "i_only": "iSPN-only",
    "left": "Left d+i",
    "right": "Right d+i",
    "8": "state 8",
    "8_to_10": "8→10",
    "4": "state 4",
    "4_to_5": "4→5",
    "state10": "Left (d+i)",
    "10_to_11": "+ Right i",
    "state5": "Right (d+i)",
    "5_to_7": "+ Left i",
}


def parse_sequence(value: Any) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(float(x)) for x in text.replace(",", " ").split()]


def compress_repeats(sequence: Iterable[int]) -> list[int]:
    out: list[int] = []
    for value in sequence:
        value = int(value)
        if not out or out[-1] != value:
            out.append(value)
    return out


def _first_index(sequence: list[int], state: int) -> float:
    try:
        return float(sequence.index(int(state)))
    except ValueError:
        return np.nan


def _first_non_source_after_first_episode(sequence: list[int], source: int) -> float:
    try:
        index = sequence.index(int(source))
    except ValueError:
        return np.nan
    if index + 1 >= len(sequence):
        return np.nan
    return float(sequence[index + 1])


def _direct_transition_count(sequence: list[int], source: int, target: int) -> int:
    return sum(
        int(sequence[index] == int(source) and sequence[index + 1] == int(target))
        for index in range(len(sequence) - 1)
    )


def _direct_transition_target_end_count(sequence: list[int], source: int, target: int) -> int:
    return sum(
        int(
            sequence[index] == int(source)
            and sequence[index + 1] == int(target)
            and index + 1 == len(sequence) - 1
        )
        for index in range(len(sequence) - 1)
    )


def _episode_counts(sequence: list[int], state: int, previous_state: int | None = None) -> tuple[int, int]:
    total = 0
    ended = 0
    for index, value in enumerate(sequence):
        if int(value) != int(state):
            continue
        if previous_state is not None and (
            index == 0 or int(sequence[index - 1]) != int(previous_state)
        ):
            continue
        total += 1
        ended += int(index == len(sequence) - 1)
    return ended, total


def _has_target_after_source(sequence: list[int], source: int, target: int) -> bool:
    for index, value in enumerate(sequence[:-1]):
        if int(value) == int(source) and int(target) in sequence[index + 1 :]:
            return True
    return False


def _network_id_from_session(session: Any) -> int | None:
    match = re.search(r"network[_-]?(\d+)", str(session), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def prepare_prediction_trials(
    trial_sequences: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Convert Notebook 05 trial walks to the exact trial table used by Fig. 5."""
    required = {"session", "trial_id", "choice_left", "decision_time_ms", "sequence"}
    missing = sorted(required.difference(trial_sequences.columns))
    if missing:
        raise KeyError(f"Missing required trial-sequence columns: {missing}")

    rows: list[dict[str, Any]] = []
    for row in trial_sequences.itertuples(index=False):
        sequence = compress_repeats(parse_sequence(row.sequence))
        choice_left = float(row.choice_left)
        decision_time_ms = float(row.decision_time_ms)

        network_id = _network_id_from_session(row.session)
        source_trial_num = getattr(row, "source_trial_num", row.trial_id)
        base: dict[str, Any] = {
            "dataset": str(dataset_name),
            "dataset_source": getattr(row, "dataset", dataset_name),
            "session": str(row.session),
            "network_id": network_id if network_id is not None else str(row.session),
            "trial_num": int(source_trial_num),
            "trial_id": int(row.trial_id),
            "trial_uid": f"{row.session}::{int(row.trial_id)}",
            "choice_raw": choice_left,
            "choice_left": choice_left,
            "reaction_time": decision_time_ms,
            "decision_time_ms": decision_time_ms,
            "log_reaction_time": float(np.log(decision_time_ms)),
            "ending_state": int(sequence[-1]),
            "state_sequence_phase0_compressed": " ".join(map(str, sequence)),
            "compressed_seq": tuple(sequence),
            "first_after_12": _first_non_source_after_first_episode(sequence, 12),
        }
        for state in STATE_IDS:
            base[f"V{state}"] = int(state in sequence)
            base[f"t{state}_first"] = _first_index(sequence, state)
        for source, target in TRANSITION_PAIRS:
            count = _direct_transition_count(sequence, source, target)
            end_count = _direct_transition_target_end_count(sequence, source, target)
            base[f"direct_{source}_to_{target}_count"] = int(count)
            base[f"direct_{source}_to_{target}"] = int(count > 0)
            base[f"direct_{source}_to_{target}_dest_end_count"] = int(end_count)
            base[f"direct_{source}_to_{target}_dest_end"] = int(end_count > 0)
        rows.append(base)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No prediction trials were built for {dataset_name}.")
    sort_columns = ["session", "trial_id"]
    return out.sort_values(sort_columns).reset_index(drop=True)


# -----------------------------------------------------------------------------
# CBGT reaction-time metadata
# -----------------------------------------------------------------------------

_RT_COLUMNS = (
    "reaction_time",
    "reaction_time_ms",
    "decisionduration",
    "decision_duration",
    "decision_time_ms",
    "decision_time",
)


def _drop_unnamed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.astype(str).str.contains(r"^Unnamed")].copy()


def _parse_seed_index(value: Any) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    match = re.search(r"(\d+)$", str(value))
    if match:
        return int(match.group(1))
    return int(value)


def _lookup_seed_run(container: Any, seed_index: int) -> Any:
    if isinstance(container, dict):
        for key in (seed_index, str(seed_index), f"seed_{seed_index}"):
            if key in container:
                return container[key]
        if len(container) == 1:
            return next(iter(container.values()))
    try:
        return container[seed_index]
    except Exception:
        if hasattr(container, "__len__") and len(container) == 1:
            return container[0]
        raise KeyError(f"Could not locate seed {seed_index} in the network pickle.")


def _lookup_decision_duration(run: Any, trial_num: int) -> float:
    if isinstance(run, dict):
        datatables = run.get("datatables")
    else:
        datatables = getattr(run, "datatables", None)
    if datatables is None:
        raise KeyError("The network run has no 'datatables' entry.")
    durations = datatables["decisionduration"]
    if isinstance(durations, pd.Series):
        if trial_num in durations.index:
            value = durations.loc[trial_num]
        else:
            value = durations.iloc[int(trial_num)]
    elif isinstance(durations, dict):
        value = durations.get(trial_num, durations.get(str(trial_num)))
    else:
        value = np.asarray(durations).reshape(-1)[int(trial_num)]
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid decision duration {value!r} for trial {trial_num}.")
    return value


def _standardized_cbgt_trial_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _drop_unnamed(frame)
    if "trial_num" not in frame.columns:
        raise KeyError("CBGT data_conf.csv is missing 'trial_num'.")
    keys = [column for column in ("seed", "trial_num") if column in frame.columns]
    sort_columns = keys + (["bin_num"] if "bin_num" in frame.columns else [])
    if sort_columns:
        frame = frame.sort_values(sort_columns).copy()
    frame["trial_id"] = frame.groupby(keys, sort=True).ngroup() + 1
    columns = ["trial_id", "trial_num"] + (["seed"] if "seed" in frame.columns else [])
    return frame[columns].drop_duplicates().sort_values("trial_id").reset_index(drop=True)


def build_cbgt_exact_trial_metadata(
    state_networks_dir: Path,
    raw_networks_dir: Path | None = None,
    output_path: Path | None = None,
    n_networks: int = 300,
) -> pd.DataFrame:
    """Build the compact exact CBGT reaction-time table used by Fig. 5C."""
    state_networks_dir = Path(state_networks_dir)
    raw_networks_dir = Path(raw_networks_dir) if raw_networks_dir is not None else state_networks_dir
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for network_id in range(1, int(n_networks) + 1):
        state_dir = state_networks_dir / f"network_{network_id}"
        raw_dir = raw_networks_dir / f"network_{network_id}"
        conf_path = state_dir / "data_conf.csv"
        if not conf_path.exists():
            failures.append(f"network_{network_id}: missing {conf_path}")
            continue
        conf = _drop_unnamed(pd.read_csv(conf_path))
        keys = _standardized_cbgt_trial_keys(conf)

        direct_rt_column = next((column for column in _RT_COLUMNS if column in conf.columns), None)
        direct_lookup: dict[int, float] = {}
        if direct_rt_column is not None:
            temp = conf.copy()
            key_columns = [column for column in ("seed", "trial_num") if column in temp.columns]
            temp = temp.sort_values(key_columns + (["bin_num"] if "bin_num" in temp.columns else []))
            temp["trial_id"] = temp.groupby(key_columns, sort=True).ngroup() + 1
            direct_lookup = (
                temp.groupby("trial_id", sort=True)[direct_rt_column]
                .first()
                .astype(float)
                .to_dict()
            )

        pickle_path = raw_dir / f"network_{network_id}.pickle"
        network_pickle = None
        if pickle_path.exists():
            with open(pickle_path, "rb") as stream:
                network_pickle = pickle.load(stream)

        for record in keys.itertuples(index=False):
            trial_id = int(record.trial_id)
            trial_num = int(record.trial_num)
            seed_value = getattr(record, "seed", None)
            if network_pickle is not None:
                seed_index = _parse_seed_index(seed_value)
                run = _lookup_seed_run(network_pickle, seed_index)
                decision_time = _lookup_decision_duration(run, trial_num)
                source = "network_pickle:datatables.decisionduration"
            elif trial_id in direct_lookup:
                decision_time = float(direct_lookup[trial_id])
                source = direct_rt_column
            else:
                failures.append(
                    f"network_{network_id}, trial_id={trial_id}: no exact RT column and missing {pickle_path}"
                )
                continue
            rows.append(
                {
                    "network_id": int(network_id),
                    "trial_id": trial_id,
                    "trial_num": trial_num,
                    "seed": seed_value,
                    "decision_time_ms": decision_time,
                    "rt_source": source,
                }
            )

    metadata = pd.DataFrame(rows)
    expected_rows = int(n_networks) * 50
    if failures or len(metadata) != expected_rows:
        preview = "\n".join(failures[:12])
        raise RuntimeError(
            f"Could not build the complete exact CBGT trial metadata: "
            f"{len(metadata)}/{expected_rows} rows.\n{preview}"
        )
    if metadata.duplicated(["network_id", "trial_id"]).any():
        raise ValueError("Duplicate network_id/trial_id rows in exact CBGT metadata.")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(output_path, index=False)
    return metadata


def apply_cbgt_exact_trial_metadata(
    trial_sequences: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    trials = trial_sequences.copy()
    trials["network_id"] = trials["session"].map(_network_id_from_session)
    if trials["network_id"].isna().any():
        raise ValueError("Could not parse every CBGT network ID from the session column.")
    trials["network_id"] = trials["network_id"].astype(int)
    metadata = metadata.copy()
    metadata["network_id"] = metadata["network_id"].astype(int)
    metadata["trial_id"] = metadata["trial_id"].astype(int)
    merged = trials.merge(
        metadata[["network_id", "trial_id", "trial_num", "decision_time_ms"]].rename(
            columns={"trial_num": "source_trial_num"}
        ),
        on=["network_id", "trial_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_approximate", "_exact"),
    )
    if merged["decision_time_ms_exact"].isna().any():
        missing = merged.loc[
            merged["decision_time_ms_exact"].isna(), ["session", "trial_id"]
        ].head()
        raise ValueError(f"Exact CBGT RT metadata did not match all trials:\n{missing}")
    merged["decision_time_ms"] = merged.pop("decision_time_ms_exact").astype(float)
    if "decision_time_ms_approximate" in merged.columns:
        merged = merged.drop(columns="decision_time_ms_approximate")
    return merged


# -----------------------------------------------------------------------------
# Predictions 1-3
# -----------------------------------------------------------------------------


def _percentile_summary(values: np.ndarray) -> tuple[float, float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan,) * 5
    return tuple(np.percentile(values, [2.5, 25, 50, 75, 97.5]))


def _result_row(
    dataset: str,
    prediction: str,
    metric: str,
    group: str,
    observed: float,
    distribution: np.ndarray,
    n_eligible: int,
    n_resampled: int,
    notes: str,
) -> dict[str, Any]:
    q02_5, q25, q50, q75, q97_5 = _percentile_summary(distribution)
    return {
        "dataset": dataset,
        "prediction": prediction,
        "metric": metric,
        "group": group,
        "observed": observed,
        "ci95_lower": q02_5,
        "ci95_upper": q97_5,
        "q02_5": q02_5,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q97_5": q97_5,
        "n_eligible_trials": int(n_eligible),
        "n_resampled_per_round": int(n_resampled),
        "B": int(len(distribution)),
        "notes": notes,
    }


def _bootstrap_binary_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or int(sample_size) <= 0:
        return np.full(N_BOOTSTRAP, np.nan)
    probability = float(values.mean())
    return rng.binomial(int(sample_size), probability, size=N_BOOTSTRAP) / float(sample_size)


def _bootstrap_ratio_from_trial_counts(
    numerator: np.ndarray,
    denominator: np.ndarray,
    rng: np.random.Generator,
    sample_size: int,
    chunk_size: int = 128,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    n_trials = len(numerator)
    if n_trials == 0 or int(sample_size) <= 0:
        return np.full(N_BOOTSTRAP, np.nan)
    output = np.empty(N_BOOTSTRAP, dtype=float)
    for start in range(0, N_BOOTSTRAP, int(chunk_size)):
        stop = min(start + int(chunk_size), N_BOOTSTRAP)
        indices = rng.integers(0, n_trials, size=(stop - start, int(sample_size)))
        numerator_sum = numerator[indices].sum(axis=1)
        denominator_sum = denominator[indices].sum(axis=1)
        output[start:stop] = np.divide(
            numerator_sum,
            denominator_sum,
            out=np.full(stop - start, np.nan),
            where=denominator_sum > 0,
        )
    return output


def _observed_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    total = float(np.asarray(denominator, dtype=float).sum())
    if total <= 0:
        return np.nan
    return float(np.asarray(numerator, dtype=float).sum() / total)


def _state_end_arrays(
    trials: pd.DataFrame,
    state: int,
    previous_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ended: list[int] = []
    total: list[int] = []
    for sequence in trials["compressed_seq"]:
        end_count, total_count = _episode_counts(list(sequence), state, previous_state)
        ended.append(end_count)
        total.append(total_count)
    return np.asarray(ended, dtype=float), np.asarray(total, dtype=float)


def _uncertainty_note(lower: float, upper: float) -> str:
    if lower >= UNCERTAINTY_BAND[0] and upper <= UNCERTAINTY_BAND[1]:
        return "CI inside uncertainty band [0.45, 0.55]"
    if upper < 0.5:
        return "CI below 0.5"
    if lower > 0.5:
        return "CI above 0.5"
    return "CI overlaps uncertainty band [0.45, 0.55]"


def _direction_note(lower: float, upper: float) -> str:
    if lower > 0.5:
        return "CI above 0.5"
    if upper < 0.5:
        return "CI below 0.5"
    return "CI crosses 0.5"


def analyze_fig5_predictions(
    trial_sequences: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Predictions 1-3 analysis."""
    trials = prepare_prediction_trials(trial_sequences, dataset_name)
    rng = np.random.default_rng(PREDICTION_RANDOM_SEED)
    result_rows: list[dict[str, Any]] = []
    distribution_columns: dict[str, np.ndarray] = {}
    count_rows: list[dict[str, Any]] = []

    # Prediction 1
    groups = {
        "GU_visited_1_2_3": trials.loc[trials[["V1", "V2", "V3"]].any(axis=1)].copy(),
        "GL_visited_8_or_10": trials.loc[trials[["V8", "V10"]].any(axis=1)].copy(),
        "GR_visited_4_or_5": trials.loc[trials[["V4", "V5"]].any(axis=1)].copy(),
    }
    balanced_n = min(len(group) for group in groups.values())
    for label, group in groups.items():
        distribution = _bootstrap_binary_mean(
            group["choice_left"].to_numpy(), rng, balanced_n
        )
        distribution_columns[f"p1_{label}_P_left"] = distribution
        lower, _, _, _, upper = _percentile_summary(distribution)
        note = _uncertainty_note(lower, upper) if label.startswith("GU") else _direction_note(lower, upper)
        result_rows.append(
            _result_row(
                dataset_name,
                "1_without_dSPN_iSPN_only_uncertainty",
                "left_choice_probability",
                label,
                float(group["choice_left"].mean()),
                distribution,
                len(group),
                balanced_n,
                note,
            )
        )
        count_rows.append(
            {
                "dataset": dataset_name,
                "prediction": "1",
                "group": label,
                "n_trials": int(len(group)),
                "n_units": int(group["network_id"].nunique()),
                "n_left_choices": int(group["choice_left"].eq(1).sum()),
                "n_right_choices": int(group["choice_left"].eq(0).sum()),
            }
        )

    # Prediction 2
    for source, target, label in (
        (8, 10, "left_channel_8_to_10"),
        (4, 5, "right_channel_4_to_5"),
    ):
        base = trials.loc[trials[f"V{source}"].eq(1)].copy()
        source_end, source_total = _state_end_arrays(base, source)
        target_end, target_total = _state_end_arrays(base, target, previous_state=source)
        source_distribution = _bootstrap_ratio_from_trial_counts(
            source_end, source_total, rng, len(base)
        )
        target_distribution = _bootstrap_ratio_from_trial_counts(
            target_end, target_total, rng, len(base)
        )
        distribution_columns[
            f"p2_{label}_P_END_after_state_{source}"
        ] = source_distribution
        distribution_columns[
            f"p2_{label}_P_END_after_state_{target}_given_direct_{source}_to_{target}"
        ] = target_distribution
        result_rows.extend(
            [
                _result_row(
                    dataset_name,
                    "2_same_channel_coactivation_terminal_phase",
                    f"P(next=END | current_state={source})",
                    label,
                    _observed_ratio(source_end, source_total),
                    source_distribution,
                    len(base),
                    len(base),
                    "CLAW-style state END probability: END outgoing events / all outgoing events for the source state",
                ),
                _result_row(
                    dataset_name,
                    "2_same_channel_coactivation_terminal_phase",
                    f"P(next=END | previous_state={source}, current_state={target})",
                    label,
                    _observed_ratio(target_end, target_total),
                    target_distribution,
                    len(base),
                    len(base),
                    f"denominator is direct compressed {source}->{target} episodes",
                ),
            ]
        )
        count_rows.append(
            {
                "dataset": dataset_name,
                "prediction": "2",
                "group": f"{label}_base_V{source}",
                "n_trials": int(len(base)),
                "n_units": int(base["network_id"].nunique()),
                f"state_{source}_episode_denominator": int(source_total.sum()),
                f"state_{source}_END_numerator": int(source_end.sum()),
                f"direct_{source}_to_{target}_episode_denominator": int(target_total.sum()),
                f"direct_{source}_to_{target}_then_END_numerator": int(target_end.sum()),
            }
        )

    # Prediction 3: raw trial distributions; no bootstrap.
    raw_frames: list[pd.DataFrame] = []
    for source, target, comparison in (
        (10, 11, "left_channel_10_then_later_11"),
        (5, 7, "right_channel_5_then_later_7"),
    ):
        source_visited = trials[f"V{source}"].eq(1)
        target_later = trials["compressed_seq"].map(
            lambda sequence: _has_target_after_source(list(sequence), source, target)
        )
        for group_name, definition, mask in (
            ("A", f"V({source})=1 and no later {target} after {source}", source_visited & ~target_later),
            ("B", f"V({source})=1 and later {target} after {source}", source_visited & target_later),
        ):
            group = trials.loc[mask].copy()
            group["prediction"] = "3_extra_opponent_iSPN_raw_decision_time_distribution"
            group["comparison"] = comparison
            group["source_state"] = int(source)
            group["dest_state"] = int(target)
            group["group"] = group_name
            group["group_definition"] = definition
            group["dest_after_source"] = target_later.loc[group.index].to_numpy(bool)
            group["direct_source_to_dest"] = group[f"direct_{source}_to_{target}"].astype(int)
            raw_frames.append(group)
            count_rows.append(
                {
                    "dataset": dataset_name,
                    "prediction": "3",
                    "comparison": comparison,
                    "source_state": int(source),
                    "dest_state": int(target),
                    "group": group_name,
                    "group_definition": definition,
                    "n_trials": int(len(group)),
                    "n_units": int(group["network_id"].nunique()),
                    "n_direct_source_to_dest": int(group["direct_source_to_dest"].sum()),
                    "n_ended_source": int(group["ending_state"].eq(source).sum()),
                    "n_ended_dest": int(group["ending_state"].eq(target).sum()),
                }
            )

    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    main_results = pd.DataFrame(result_rows)
    bootstrap = pd.DataFrame(distribution_columns)
    counts = pd.DataFrame(count_rows)
    return main_results, bootstrap, raw, counts, trials


# -----------------------------------------------------------------------------
# Statistical tests used for Fig. 5 annotations
# -----------------------------------------------------------------------------


def _clean_numeric(values: Any) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return array[np.isfinite(array)]


def _bootstrap_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.array([], dtype=float)
    return _clean_numeric(frame[column])


def _raw_rt_values(frame: pd.DataFrame, plot_key: str) -> np.ndarray:
    mapping = {
        "state10": ("left_channel_10_then_later_11", "A"),
        "10_to_11": ("left_channel_10_then_later_11", "B"),
        "state5": ("right_channel_5_then_later_7", "A"),
        "5_to_7": ("right_channel_5_then_later_7", "B"),
    }
    comparison, group = mapping[plot_key]
    values = frame.loc[
        frame["comparison"].eq(comparison) & frame["group"].eq(group),
        "reaction_time",
    ]
    return _clean_numeric(values)


def _p_to_stars(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "n.s."


def _two_sample_ttest_exact(group1: np.ndarray, group2: np.ndarray) -> dict[str, float | int]:
    group1 = _clean_numeric(group1)
    group2 = _clean_numeric(group2)
    output: dict[str, float | int] = {
        "n_group1": int(len(group1)),
        "n_group2": int(len(group2)),
        "mean_group1": float(group1.mean()) if len(group1) else np.nan,
        "mean_group2": float(group2.mean()) if len(group2) else np.nan,
        "mean_difference_group2_minus_group1": (
            float(group2.mean() - group1.mean()) if len(group1) and len(group2) else np.nan
        ),
        "t_statistic": np.nan,
        "df": np.nan,
        "p_value": np.nan,
    }
    if len(group1) < 2 or len(group2) < 2:
        return output
    # The source plotting notebook used equal_var=True.  Preserve it exactly.
    result = stats.ttest_ind(
        group1,
        group2,
        equal_var=True,
        nan_policy="omit",
        alternative="two-sided",
    )
    output["t_statistic"] = float(result.statistic)
    output["p_value"] = float(result.pvalue)
    df = getattr(result, "df", len(group1) + len(group2) - 2)
    output["df"] = float(df)
    return output


def run_fig5_significance_tests(
    cbgt_bootstrap: pd.DataFrame,
    empirical_bootstrap: pd.DataFrame,
    cbgt_raw: pd.DataFrame,
    empirical_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anova_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    bootstrap_sources = {
        "CBGT": cbgt_bootstrap,
        "IBL+Steinmetz": empirical_bootstrap,
    }
    raw_sources = {
        "CBGT": cbgt_raw,
        "IBL+Steinmetz": empirical_raw,
    }

    for source, frame in bootstrap_sources.items():
        groups = {
            key: _bootstrap_values(frame, column)
            for key, column in PANEL_A_BOOTSTRAP_COLUMNS.items()
        }
        result = stats.f_oneway(*groups.values())
        n_total = sum(len(values) for values in groups.values())
        anova_rows.append(
            {
                "panel": "A",
                "source": source,
                "comparison": "iSPN-only vs Left d+i vs Right d+i",
                "test": "one-way ANOVA",
                "data_level": "bootstrap distribution",
                "n_groups": 3,
                "n_total": int(n_total),
                "df_between": 2,
                "df_within": int(n_total - 3),
                "F_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
        for group1, group2 in PANEL_PAIRWISE_COMPARISONS["A"]:
            test = _two_sample_ttest_exact(groups[group1], groups[group2])
            adjusted = min(float(test["p_value"]) * 2.0, 1.0)
            pairwise_rows.append(
                {
                    "panel": "A",
                    "source": source,
                    "comparison": f"{PLOT_KEY_LABELS[group1]} vs {PLOT_KEY_LABELS[group2]}",
                    "group1": group1,
                    "group2": group2,
                    "group1_label": PLOT_KEY_LABELS[group1],
                    "group2_label": PLOT_KEY_LABELS[group2],
                    "test": "post hoc two-sample t test after one-way ANOVA",
                    "alternative": "two-sided",
                    "data_level": "bootstrap distribution",
                    "correction": "Bonferroni across two Panel A comparisons within source",
                    "bonferroni_multiplier": 2,
                    **test,
                    "p_adjusted": adjusted,
                    "p_for_stars": adjusted,
                    "significance": _p_to_stars(adjusted),
                }
            )

    for source, frame in bootstrap_sources.items():
        groups = {
            key: _bootstrap_values(frame, column)
            for key, column in PANEL_B_BOOTSTRAP_COLUMNS.items()
        }
        for group1, group2 in PANEL_PAIRWISE_COMPARISONS["B"]:
            test = _two_sample_ttest_exact(groups[group1], groups[group2])
            pairwise_rows.append(
                {
                    "panel": "B",
                    "source": source,
                    "comparison": f"{PLOT_KEY_LABELS[group1]} vs {PLOT_KEY_LABELS[group2]}",
                    "group1": group1,
                    "group2": group2,
                    "group1_label": PLOT_KEY_LABELS[group1],
                    "group2_label": PLOT_KEY_LABELS[group2],
                    "test": "planned two-sample t test",
                    "alternative": "two-sided",
                    "data_level": "bootstrap distribution",
                    "correction": "none",
                    "bonferroni_multiplier": np.nan,
                    **test,
                    "p_adjusted": np.nan,
                    "p_for_stars": test["p_value"],
                    "significance": _p_to_stars(float(test["p_value"])),
                }
            )

    for source, frame in raw_sources.items():
        groups = {key: _raw_rt_values(frame, key) for key in ("state10", "10_to_11", "state5", "5_to_7")}
        for group1, group2 in PANEL_PAIRWISE_COMPARISONS["C"]:
            test = _two_sample_ttest_exact(groups[group1], groups[group2])
            pairwise_rows.append(
                {
                    "panel": "C",
                    "source": source,
                    "comparison": f"{PLOT_KEY_LABELS[group1]} vs {PLOT_KEY_LABELS[group2]}",
                    "group1": group1,
                    "group2": group2,
                    "group1_label": PLOT_KEY_LABELS[group1],
                    "group2_label": PLOT_KEY_LABELS[group2],
                    "test": "planned two-sample t test",
                    "alternative": "two-sided",
                    "data_level": "raw reaction_time trial distribution",
                    "correction": "none",
                    "bonferroni_multiplier": np.nan,
                    **test,
                    "p_adjusted": np.nan,
                    "p_for_stars": test["p_value"],
                    "significance": _p_to_stars(float(test["p_value"])),
                }
            )

    return pd.DataFrame(anova_rows), pd.DataFrame(pairwise_rows)


def standardized_fig5_summary(
    cbgt_main: pd.DataFrame,
    empirical_main: pd.DataFrame,
    cbgt_raw: pd.DataFrame,
    empirical_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Return the compact table consumed by other repository utilities."""
    rows: list[dict[str, Any]] = []
    for dataset, main, raw in (
        ("CBGT", cbgt_main, cbgt_raw),
        ("Empirical", empirical_main, empirical_raw),
    ):
        panel_a = {
            "GU_visited_1_2_3": "iSPN-only",
            "GL_visited_8_or_10": "Left d(+i)",
            "GR_visited_4_or_5": "Right d(+i)",
        }
        for source_group, plot_group in panel_a.items():
            row = main.loc[main["group"].eq(source_group)].iloc[0]
            rows.append({"dataset": dataset, "panel": "A", "group": plot_group, **row[["observed", "q02_5", "q25", "q50", "q75", "q97_5"]].to_dict()})

        panel_b_specs = (
            ("current_state=8", "Left d"),
            ("previous_state=8", "Left d(+i)"),
            ("current_state=4", "Right d"),
            ("previous_state=4", "Right d(+i)"),
        )
        for token, plot_group in panel_b_specs:
            row = main.loc[main["metric"].astype(str).str.contains(token, regex=False)].iloc[0]
            rows.append({"dataset": dataset, "panel": "B", "group": plot_group, **row[["observed", "q02_5", "q25", "q50", "q75", "q97_5"]].to_dict()})

        raw_specs = (
            ("left_channel_10_then_later_11", "A", "Left d(+i)"),
            ("left_channel_10_then_later_11", "B", "Left d(+i)\n+ Right i"),
            ("right_channel_5_then_later_7", "A", "Right d(+i)"),
            ("right_channel_5_then_later_7", "B", "Right d(+i)\n+ Left i"),
        )
        for comparison, group, plot_group in raw_specs:
            values = raw.loc[
                raw["comparison"].eq(comparison) & raw["group"].eq(group),
                "reaction_time",
            ].to_numpy(float)
            q02_5, q25, q50, q75, q97_5 = _percentile_summary(values)
            rows.append(
                {
                    "dataset": dataset,
                    "panel": "C",
                    "group": plot_group,
                    "observed": float(values.mean()),
                    "q02_5": q02_5,
                    "q25": q25,
                    "q50": q50,
                    "q75": q75,
                    "q97_5": q97_5,
                }
            )
    return pd.DataFrame(rows)


def long_bootstrap_table(
    cbgt_bootstrap: pd.DataFrame,
    empirical_bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    reverse = {**{value: ("A", key) for key, value in PANEL_A_BOOTSTRAP_COLUMNS.items()}, **{value: ("B", key) for key, value in PANEL_B_BOOTSTRAP_COLUMNS.items()}}
    rows: list[pd.DataFrame] = []
    for dataset, frame in (("CBGT", cbgt_bootstrap), ("Empirical", empirical_bootstrap)):
        for column in frame.columns:
            panel, key = reverse[column]
            part = pd.DataFrame(
                {
                    "dataset": dataset,
                    "panel": panel,
                    "group": PLOT_KEY_LABELS[key],
                    "replicate": np.arange(len(frame), dtype=int),
                    "value": frame[column].to_numpy(float),
                }
            )
            rows.append(part)
    return pd.concat(rows, ignore_index=True)