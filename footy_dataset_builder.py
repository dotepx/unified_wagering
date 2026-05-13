from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from feature_building import (
    QuantileBucketSpec,
    add_calendar_categories,
    add_ewm_features,
    add_positive_quantile_target_buckets,
)
from utilities import (
    finalize_grammar_frame,
    load_parquet,
    make_save_to_parser,
    save_parquet,
    smart_type,
    update_status,
)


__all__: list[str] = []

if __name__ not in {"__main__", "__mp_main__"}:
    raise ImportError(
        "footy_dataset_builder.py is a script entrypoint "
        "and should not be imported"
    )

# ---------------------------------------------------------------------------
# Source column candidates (footy_loader grammar output)
# ---------------------------------------------------------------------------

_HOME_GOALS: Final[tuple[str, ...]] = (
    "FEATURE_HOMEGOALCOUNT",
    "FEATURE_HOME_GOAL_COUNT",
    "FEATURE_HOME_GOALS",
)
_AWAY_GOALS: Final[tuple[str, ...]] = (
    "FEATURE_AWAYGOALCOUNT",
    "FEATURE_AWAY_GOAL_COUNT",
    "FEATURE_AWAY_GOALS",
)
_HOME_HT: Final[tuple[str, ...]] = (
    "FEATURE_HT_GOALS_TEAM_A",
    "FEATURE_HOMEGOALCOUNT_HT",
    "FEATURE_HOME_HT_GOALS",
)
_AWAY_HT: Final[tuple[str, ...]] = (
    "FEATURE_HT_GOALS_TEAM_B",
    "FEATURE_AWAYGOALCOUNT_HT",
    "FEATURE_AWAY_HT_GOALS",
)
_HOME_NAME: Final[tuple[str, ...]] = ("SERIES_HOME",)
_AWAY_NAME: Final[tuple[str, ...]] = ("SERIES_AWAY",)
_HOME_ID: Final[tuple[str, ...]] = ("SERIES_HOME_ID",)
_AWAY_ID: Final[tuple[str, ...]] = ("SERIES_AWAY_ID",)
_STATUS: Final[tuple[str, ...]] = ("CATEGORY_STATUS",)
_MATCH_ID: Final[tuple[str, ...]] = ("LABEL_ID", "LABEL_MATCH_ID")
_DATE_UNIX: Final[tuple[str, ...]] = ("LABEL_DATE_UNIX",)
_NEUTRAL: Final[tuple[str, ...]] = (
    "IS_NO_HOME_AWAY",
    "FEATURE_NO_HOME_AWAY",
    "IS_NEUTRAL",
)

# ---------------------------------------------------------------------------
# Output column names
# ---------------------------------------------------------------------------

_LABEL_DATE: Final[str] = "LABEL_DATE"
_LABEL_MATCHID: Final[str] = "LABEL_MATCHID"
_LABEL_COMPLETE: Final[str] = "LABEL_COMPLETE"

_SERIES_TEAM: Final[str] = "SERIES_TEAM"
_SERIES_TEAM_ID: Final[str] = "SERIES_TEAM_ID"
_SERIES_CONTRA: Final[str] = "SERIES_CONTRA"
_SERIES_CONTRA_ID: Final[str] = "SERIES_CONTRA_ID"

_FEATURE_MADE: Final[str] = "FEATURE_MADE_SCORE"
_FEATURE_GAVE: Final[str] = "FEATURE_GAVE_SCORE"
_FEATURE_DELTA: Final[str] = "FEATURE_DELTA_SCORE"
_FEATURE_TOTAL: Final[str] = "FEATURE_TOTAL_SCORE"
_FEATURE_WIN: Final[str] = "FEATURE_WIN"
_FEATURE_DRAW: Final[str] = "FEATURE_DRAW"
_FEATURE_CLEAN: Final[str] = "FEATURE_CLEAN_SHEET"
_FEATURE_SCORED: Final[str] = "FEATURE_SCORED"

TARGET_DELTA_SCORE: Final[str] = "TARGET_DELTA_SCORE"
TARGET_TOTAL_SCORE: Final[str] = "TARGET_TOTAL_SCORE"
TARGET_MADE_SCORE: Final[str] = "TARGET_MADE_SCORE"

_BASE_EWM_COLS: Final[tuple[str, ...]] = (
    _FEATURE_WIN,
    _FEATURE_DRAW,
    _FEATURE_MADE,
    _FEATURE_GAVE,
    _FEATURE_DELTA,
    _FEATURE_TOTAL,
    _FEATURE_CLEAN,
    _FEATURE_SCORED,
)
_FOOTY_ID_KEY: Final[list[str]] = [_SERIES_TEAM_ID, _LABEL_DATE]

# ---------------------------------------------------------------------------
# Column spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _FootyColumnSpec:
    """hold detected source column names from a footy_loader parquet"""

    home_goals: str
    away_goals: str
    home_name: str
    away_name: str
    home_id: str
    away_id: str
    status_col: str
    match_id_col: str
    date_unix_col: str
    home_ht: str | None
    away_ht: str | None
    is_neutral_col: str | None
    stat_pairs: tuple[tuple[str, str, str], ...]


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AppConfig:
    """hold runtime settings"""

    data: Path
    save_to: Path
    positive_bucket_count: int = 9
    short_ewm_span: int = 6
    long_ewm_span: int = 16

    def __post_init__(self) -> None:
        if not self.data.exists():
            raise ValueError(f"data does not exist: {self.data}")
        if self.save_to.name in {"", ".", ".."}:
            raise ValueError("save_to must include an output filename")
        if self.save_to.suffix.lower() not in {".pqt", ".parquet"}:
            raise ValueError("save_to must end with .pqt or .parquet")
        if self.positive_bucket_count <= 0:
            raise ValueError(
                "positive_bucket_count must be positive; "
                f"received {self.positive_bucket_count}"
            )
        if self.short_ewm_span <= 0 or self.long_ewm_span <= 0:
            raise ValueError("ewm spans must be positive")
        if self.short_ewm_span >= self.long_ewm_span:
            raise ValueError(
                "short_ewm_span must be less than long_ewm_span"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _require(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    label: str,
) -> str:
    col: str | None = _find(df, candidates)
    if col is None:
        raise ValueError(
            f"required column {label!r} not found; tried: {list(candidates)}"
        )
    return col


def _detect_stat_pairs(
    df: pd.DataFrame,
) -> tuple[tuple[str, str, str], ...]:
    """find FEATURE_*_TEAM_A / FEATURE_*_TEAM_B pairs."""
    pairs: list[tuple[str, str, str]] = []
    col: str
    for col in sorted(df.columns):
        if not col.startswith("FEATURE_") or "TEAM_A" not in col:
            continue
        contra: str = col.replace("TEAM_A", "TEAM_B")
        if contra not in df.columns:
            continue
        clean: str = (
            col.removeprefix("FEATURE_")
            .replace("_TEAM_A", "")
            .replace("TEAM_A_", "")
            .strip("_")
        )
        if clean:
            pairs.append((col, contra, clean))
    return tuple(pairs)


def _detect_column_spec(df: pd.DataFrame) -> _FootyColumnSpec:
    return _FootyColumnSpec(
        home_goals=_require(df, _HOME_GOALS, "home goals"),
        away_goals=_require(df, _AWAY_GOALS, "away goals"),
        home_name=_require(df, _HOME_NAME, "home team name"),
        away_name=_require(df, _AWAY_NAME, "away team name"),
        home_id=_require(df, _HOME_ID, "home team id"),
        away_id=_require(df, _AWAY_ID, "away team id"),
        status_col=_require(df, _STATUS, "match status"),
        match_id_col=_require(df, _MATCH_ID, "match id"),
        date_unix_col=_require(df, _DATE_UNIX, "date unix"),
        home_ht=_find(df, _HOME_HT),
        away_ht=_find(df, _AWAY_HT),
        is_neutral_col=_find(df, _NEUTRAL),
        stat_pairs=_detect_stat_pairs(df),
    )


def _require_columns(
    df: pd.DataFrame,
    columns: list[str],
    frame_name: str,
) -> None:
    missing: list[str] = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {missing}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _load_footy_frame(path: Path) -> pd.DataFrame:
    df: pd.DataFrame = load_parquet(path)
    update_status(f"loaded footy frame {path}: {df.shape}")
    return df


def _build_match_dates(
    df: pd.DataFrame,
    spec: _FootyColumnSpec,
) -> pd.DataFrame:
    output: pd.DataFrame = df.copy()
    output[_LABEL_DATE] = pd.to_datetime(
        output[spec.date_unix_col], unit="s", errors="coerce"
    ).dt.normalize()
    return output


def _filter_active_matches(
    df: pd.DataFrame,
    spec: _FootyColumnSpec,
) -> pd.DataFrame:
    """keep completed matches and upcoming fixtures within 7 days."""
    output: pd.DataFrame = df.copy()
    is_complete: pd.Series = output[spec.status_col].str.lower().eq(
        "complete"
    )
    output[_LABEL_COMPLETE] = is_complete.astype("uint8")
    today: pd.Timestamp = pd.Timestamp.now().normalize()
    cutoff: pd.Timestamp = today + pd.Timedelta(days=7)
    keep: pd.Series = is_complete | output[_LABEL_DATE].between(
        today, cutoff
    )
    return output.loc[keep].copy()


def _pivot_perspective(
    df: pd.DataFrame,
    spec: _FootyColumnSpec,
    is_home: bool,
) -> pd.DataFrame:
    """build one team-perspective half (home or away)."""
    if is_home:
        team_name, team_id = spec.home_name, spec.home_id
        contra_name, contra_id = spec.away_name, spec.away_id
        made_goals, gave_goals = spec.home_goals, spec.away_goals
        ht_made, ht_gave = spec.home_ht, spec.away_ht
        stat_made_idx, stat_gave_idx = 0, 1
    else:
        team_name, team_id = spec.away_name, spec.away_id
        contra_name, contra_id = spec.home_name, spec.home_id
        made_goals, gave_goals = spec.away_goals, spec.home_goals
        ht_made, ht_gave = spec.away_ht, spec.home_ht
        stat_made_idx, stat_gave_idx = 1, 0

    out: pd.DataFrame = df.copy()
    out[_SERIES_TEAM] = out[team_name]
    out[_SERIES_TEAM_ID] = out[team_id].astype("uint64")
    out[_SERIES_CONTRA] = out[contra_name]
    out[_SERIES_CONTRA_ID] = out[contra_id].astype("uint64")
    out["IS_HOME"] = np.uint8(1 if is_home else 0)
    out["_MADE"] = pd.to_numeric(out[made_goals], errors="coerce")
    out["_GAVE"] = pd.to_numeric(out[gave_goals], errors="coerce")

    if ht_made is not None and ht_gave is not None:
        out["_MADE_HT"] = pd.to_numeric(out[ht_made], errors="coerce")
        out["_GAVE_HT"] = pd.to_numeric(out[ht_gave], errors="coerce")

    stat_col_pair: tuple[str, str, str]
    for stat_col_pair in spec.stat_pairs:
        h_col, a_col, clean = stat_col_pair
        pair = (h_col, a_col)
        out[f"_MADE_{clean}"] = pd.to_numeric(
            out[pair[stat_made_idx]], errors="coerce"
        )
        out[f"_GAVE_{clean}"] = pd.to_numeric(
            out[pair[stat_gave_idx]], errors="coerce"
        )
    return out


def _pivot_to_team_perspective(
    df: pd.DataFrame,
    spec: _FootyColumnSpec,
) -> pd.DataFrame:
    """create two rows per match: one per team perspective."""
    home: pd.DataFrame = _pivot_perspective(df, spec, is_home=True)
    away: pd.DataFrame = _pivot_perspective(df, spec, is_home=False)
    output: pd.DataFrame = pd.concat(
        [home, away], ignore_index=True
    )
    output[_LABEL_MATCHID] = output[spec.match_id_col].astype(str)
    return output.sort_values(
        _FOOTY_ID_KEY, kind="stable", ignore_index=True
    )


def _add_match_features(
    df: pd.DataFrame,
    spec: _FootyColumnSpec,
) -> pd.DataFrame:
    """add per-match FEATURE_ columns from pivoted scores.

    raw match stats are NaN for incomplete rows so they never leak
    into the model as predictive features for upcoming fixtures.
    they are present in the history tensor for completed rows.
    """
    output: pd.DataFrame = df.copy()
    complete: pd.Series = output[_LABEL_COMPLETE].eq(1)

    made: pd.Series = output["_MADE"].where(complete)
    gave: pd.Series = output["_GAVE"].where(complete)

    output[_FEATURE_MADE] = made
    output[_FEATURE_GAVE] = gave
    output[_FEATURE_DELTA] = (made - gave).astype("float64")
    output[_FEATURE_TOTAL] = (made + gave).astype("float64")
    output[_FEATURE_WIN] = (
        output[_FEATURE_DELTA].gt(0).where(complete).astype("float64")
    )
    output[_FEATURE_DRAW] = (
        output[_FEATURE_DELTA].eq(0).where(complete).astype("float64")
    )
    output[_FEATURE_CLEAN] = (
        gave.le(0).where(complete).astype("float64")
    )
    output[_FEATURE_SCORED] = (
        made.gt(0).where(complete).astype("float64")
    )

    if "_MADE_HT" in output.columns:
        ht_made: pd.Series = output["_MADE_HT"].where(complete)
        ht_gave: pd.Series = output["_GAVE_HT"].where(complete)
        output["FEATURE_MADE_HT_SCORE"] = ht_made
        output["FEATURE_GAVE_HT_SCORE"] = ht_gave
        ht_delta: pd.Series = (ht_made - ht_gave).astype("float64")
        output["FEATURE_HT_DELTA_SCORE"] = ht_delta
        output["FEATURE_HT_WIN"] = (
            ht_delta.gt(0).where(complete).astype("float64")
        )
        output["FEATURE_HT_DRAW"] = (
            ht_delta.eq(0).where(complete).astype("float64")
        )
        output["FEATURE_2H_DELTA_SCORE"] = (
            output[_FEATURE_DELTA] - ht_delta
        ).astype("float64")

    clean: str
    for _, _, clean in spec.stat_pairs:
        mc: str = f"_MADE_{clean}"
        gc: str = f"_GAVE_{clean}"
        if mc not in output.columns or gc not in output.columns:
            continue
        ms: pd.Series = output[mc].where(complete)
        gs: pd.Series = output[gc].where(complete)
        output[f"FEATURE_MVG_{clean}"] = (ms - gs).astype("float64")

    return output


def _add_rolling_features(
    df: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    """add EWM rolling features per team with no look-ahead bias."""
    optional_ewm: list[str] = [
        c
        for c in [
            "FEATURE_HT_WIN",
            "FEATURE_HT_DRAW",
            "FEATURE_HT_DELTA_SCORE",
            "FEATURE_2H_DELTA_SCORE",
        ]
        if c in df.columns
    ]
    mvg_cols: list[str] = [
        c for c in df.columns if c.startswith("FEATURE_MVG_")
    ]
    all_source: list[str] = (
        list(_BASE_EWM_COLS) + optional_ewm + mvg_cols
    )
    return add_ewm_features(
        df,
        group_col=_SERIES_TEAM_ID,
        source_columns=[c for c in all_source if c in df.columns],
        short_span=config.short_ewm_span,
        long_span=config.long_ewm_span,
        date_col=_LABEL_DATE,
    )


def _add_match_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """add days since last match and cumulative season match count."""
    output: pd.DataFrame = df.sort_values(
        _FOOTY_ID_KEY, kind="stable"
    ).copy()

    last_date: pd.Series = output.groupby(
        _SERIES_TEAM_ID, sort=False
    )[_LABEL_DATE].shift(1)
    output["FEATURE_DAYS_SINCE_LAST_MATCH"] = (
        output[_LABEL_DATE] - last_date
    ).dt.days.astype("float64")

    if "CATEGORY_SEASON" in output.columns:
        output["FEATURE_SEASON_MATCH_COUNT"] = output.groupby(
            [_SERIES_TEAM_ID, "CATEGORY_SEASON"], sort=False
        ).cumcount().astype("float64")

    return output


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """add raw TARGET_ columns from match features (NaN for incomplete).

    target names are sport-agnostic so the same downstream model and
    bucket scheme works for tennis and ncaamb:
      TARGET_DELTA_SCORE  — made minus gave (score differential)
      TARGET_TOTAL_SCORE  — made plus gave (total combined score)
      TARGET_MADE_SCORE   — per-team score
    """
    output: pd.DataFrame = df.copy()
    output[TARGET_DELTA_SCORE] = output[_FEATURE_DELTA]
    output[TARGET_TOTAL_SCORE] = output[_FEATURE_TOTAL]
    output[TARGET_MADE_SCORE] = output[_FEATURE_MADE]
    return output


def _add_bucket_targets(
    df: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    """apply quantile bucketing to each target.

    bucket 0 is always the non-positive class (LTE_0):
      TARGET_DELTA_SCORE  — LTE_0 covers losses and draws; positive
                            buckets cover wins by increasing margin.
                            downstream: P(WIN) = 1 - P(bucket_0).
      TARGET_TOTAL_SCORE  — LTE_0 covers 0-0 draws; positive buckets
                            cover all scoring levels.
      TARGET_MADE_SCORE   — LTE_0 covers goalless efforts; positive
                            buckets cover scoring levels.
    """
    return add_positive_quantile_target_buckets(
        df,
        [TARGET_DELTA_SCORE, TARGET_TOTAL_SCORE, TARGET_MADE_SCORE],
        QuantileBucketSpec(config.positive_bucket_count),
    )


def _drop_source_columns(
    df: pd.DataFrame,
    spec: _FootyColumnSpec,
    source_feature_cols: set[str],
) -> pd.DataFrame:
    """remove source columns superseded by canonical pipeline columns.

    all original FEATURE_* columns from the raw parquet are dropped
    since they are home-team-biased after the perspective pivot and
    have been replaced by FEATURE_MADE_*, FEATURE_GAVE_*, and their
    EWM derivatives.  non-FEATURE_ source identifiers are dropped too.
    """
    non_feature: set[str] = {
        spec.home_name,
        spec.away_name,
        spec.home_id,
        spec.away_id,
        spec.status_col,
        spec.match_id_col,
        spec.date_unix_col,
        "LABEL_SOURCE_HOME_ID",
        "LABEL_SOURCE_AWAY_ID",
    }
    if spec.is_neutral_col:
        non_feature.add(spec.is_neutral_col)

    to_drop: set[str] = source_feature_cols | non_feature
    existing: list[str] = [c for c in to_drop if c in df.columns]
    internal: list[str] = [c for c in df.columns if c.startswith("_")]
    return df.drop(columns=existing + internal)


def _validate(
    df: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    """validate the footy dataset output contract."""
    _require_columns(
        df,
        [
            _SERIES_TEAM_ID,
            _SERIES_CONTRA_ID,
            _LABEL_DATE,
            _LABEL_MATCHID,
            _LABEL_COMPLETE,
            _FEATURE_MADE,
            _FEATURE_GAVE,
            TARGET_DELTA_SCORE,
            TARGET_TOTAL_SCORE,
            TARGET_MADE_SCORE,
        ],
        "footy dataset",
    )
    required_buckets: list[str] = [
        f"TARGET_TYPE_{t.removeprefix('TARGET_')}"
        for t in (TARGET_DELTA_SCORE, TARGET_TOTAL_SCORE, TARGET_MADE_SCORE)
    ]
    _require_columns(df, required_buckets, "footy dataset bucket types")
    if df.duplicated([_SERIES_TEAM_ID, _LABEL_DATE, _LABEL_MATCHID]).any():
        raise ValueError(
            "footy dataset contains duplicate (team, date, match) rows"
        )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> Namespace:
    """parse cli arguments"""
    parser: ArgumentParser = make_save_to_parser(
        description="build a normalized footy feature and target dataset",
        save_to_help="output parquet path for the footy dataset",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="normalized footy parquet from footy_loader",
    )
    parser.add_argument(
        "--positive_bucket_count",
        type=int,
        default=9,
        help=(
            "positive target buckets per target; one LTE_0 bucket is "
            "always added making total = positive_bucket_count + 1"
        ),
    )
    parser.add_argument(
        "--short_ewm_span",
        type=int,
        default=6,
        help="ewm span for short-window rolling features",
    )
    parser.add_argument(
        "--long_ewm_span",
        type=int,
        default=16,
        help="ewm span for long-window rolling features",
    )
    return parser.parse_args()


def build_app_config(args: Namespace) -> AppConfig:
    """build the app config"""
    return AppConfig(
        data=args.data,
        save_to=args.save_to,
        positive_bucket_count=args.positive_bucket_count,
        short_ewm_span=args.short_ewm_span,
        long_ewm_span=args.long_ewm_span,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """run the footy dataset builder"""
    config: AppConfig = build_app_config(parse_arguments())

    update_status(f"loading footy data from {config.data}")
    raw_df: pd.DataFrame = _load_footy_frame(config.data)
    source_feature_cols: set[str] = {
        c for c in raw_df.columns if c.startswith("FEATURE_")
    }
    spec: _FootyColumnSpec = _detect_column_spec(raw_df)

    update_status("building match dates and filtering active fixtures")
    df: pd.DataFrame = _build_match_dates(raw_df, spec)
    df = _filter_active_matches(df, spec)

    n_complete: int = int(df[_LABEL_COMPLETE].sum())
    n_total: int = len(df)
    update_status(
        f"pivoting {n_total} matches ({n_complete} complete) "
        f"to team perspective"
    )
    df = _pivot_to_team_perspective(df, spec)

    update_status("computing match features")
    df = _add_match_features(df, spec)

    update_status(
        f"computing ewm rolling features "
        f"(short={config.short_ewm_span}, long={config.long_ewm_span})"
    )
    df = _add_rolling_features(df, config)
    df = _add_match_count_features(df)

    update_status("adding calendar categories and targets")
    df = add_calendar_categories(df, date_column=_LABEL_DATE)
    df = _add_targets(df)
    df = _add_bucket_targets(df, config)

    update_status("cleaning source columns and validating")
    df = _drop_source_columns(df, spec, source_feature_cols)
    df = _validate(df, config)
    df = smart_type(df)
    df = finalize_grammar_frame(df)

    update_status(f"saving footy dataset to {config.save_to}")
    save_parquet(df, config.save_to)
    update_status(
        f"saved footy dataset to {config.save_to}: {df.shape}"
    )


if __name__ == "__main__":
    main()
