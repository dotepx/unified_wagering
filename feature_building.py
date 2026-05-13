from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from market_data import LABEL_DATE


@dataclass(frozen=True, slots=True)
class QuantileBucketSpec:
    """hold target bucket settings"""

    positive_bucket_count: int = 9
    nonpositive_class: str = "LTE_0"
    positive_class_prefix: str = "POS_Q"

    def __post_init__(self) -> None:
        if self.positive_bucket_count <= 0:
            raise ValueError(
                "positive_bucket_count must be positive; "
                f"received {self.positive_bucket_count}"
            )


def make_column_token(value: str) -> str:
    """normalize a value for safe column-name embedding"""
    token: str = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().upper())
    return token.strip("_") or "UNKNOWN"


def make_log_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """compute a guarded log ratio"""
    valid: pd.Series = numer.gt(0) & denom.gt(0)
    output: pd.Series = pd.Series(np.nan, index=numer.index, dtype="float64")
    output.loc[valid] = np.log(numer.loc[valid] / denom.loc[valid])
    return output


def add_calendar_categories(
    df: pd.DataFrame,
    date_column: str = LABEL_DATE,
) -> pd.DataFrame:
    """add reusable calendar category columns"""
    if date_column not in df.columns:
        raise ValueError(f"missing date column {date_column!r}")
    output: pd.DataFrame = df.copy()
    dates: pd.Series = pd.to_datetime(output[date_column], errors="raise")
    iso_calendar: pd.DataFrame = dates.dt.isocalendar()
    output["CATEGORY_YEAR"] = dates.dt.year.astype("uint16")
    output["CATEGORY_MONTH"] = dates.dt.month.astype("uint8")
    output["CATEGORY_ISO_WEEK"] = iso_calendar["week"].astype("uint8")
    return output


def positive_quantile_edges(
    series: pd.Series,
    positive_bucket_count: int,
) -> np.ndarray:
    """compute bucket edges from positive target values"""
    positive: pd.Series = series.loc[series.gt(0)].dropna().astype("float64")
    if positive.empty:
        return np.array([], dtype="float64")
    quantiles: np.ndarray = np.linspace(
        0.0,
        1.0,
        positive_bucket_count + 1,
        dtype="float64",
    )[1:-1]
    return np.quantile(positive.to_numpy(dtype="float64"), quantiles)


def target_suffix(target_column: str) -> str:
    """return the grammar suffix for a target column"""
    if not target_column.startswith("TARGET_"):
        raise ValueError(
            f"target column must start with TARGET_: {target_column}"
        )
    suffix: str = target_column.removeprefix("TARGET_")
    if not suffix:
        raise ValueError(f"target column has empty suffix: {target_column}")
    return suffix


def add_positive_quantile_target_buckets(
    df: pd.DataFrame,
    target_columns: Sequence[str],
    spec: QuantileBucketSpec,
) -> pd.DataFrame:
    """add nonpositive and positive-quantile target buckets"""
    output: pd.DataFrame = df.copy()
    for target_column in target_columns:
        if target_column not in output.columns:
            raise ValueError(f"missing target column {target_column!r}")
        suffix: str = target_suffix(target_column)
        edges: np.ndarray = positive_quantile_edges(
            output[target_column],
            spec.positive_bucket_count,
        )
        values: np.ndarray = output[target_column].to_numpy(dtype="float64")
        valid_mask: np.ndarray = np.isfinite(values)
        nonpositive_mask: np.ndarray = valid_mask & (values <= 0.0)
        positive_mask: np.ndarray = valid_mask & (values > 0.0)
        type_values: np.ndarray = np.full(len(output), -1, dtype=np.int16)
        type_values[nonpositive_mask] = 0
        type_values[positive_mask] = (
            np.searchsorted(edges, values[positive_mask], side="right") + 1
        )
        type_values[positive_mask] = np.minimum(
            type_values[positive_mask],
            spec.positive_bucket_count,
        )

        class_values: np.ndarray = np.full(len(output), None, dtype=object)
        class_values[nonpositive_mask] = spec.nonpositive_class
        for bucket_num in range(1, spec.positive_bucket_count + 1):
            class_values[type_values == bucket_num] = (
                f"{spec.positive_class_prefix}{bucket_num:02d}"
            )

        output[f"TARGET_TYPE_{suffix}"] = pd.Series(
            type_values,
            index=output.index,
            dtype="int16",
        )
        output[f"TARGET_CLASS_{suffix}"] = pd.Series(
            class_values,
            index=output.index,
            dtype=object,
        )
        for bucket_idx in range(spec.positive_bucket_count + 1):
            bucket_column: str = f"TARGET_BUCKET_{suffix}_{bucket_idx:02d}"
            output[bucket_column] = (
                valid_mask & (type_values == bucket_idx)
            ).astype("uint8")
    return output


def add_ewm_features(
    df: pd.DataFrame,
    group_col: str,
    source_columns: Sequence[str],
    short_span: int = 6,
    long_span: int = 16,
    date_col: str = LABEL_DATE,
) -> pd.DataFrame:
    """add short and long ewm rolling averages per group, no look-ahead bias.

    output columns are named {source_col}_EWM_S and {source_col}_EWM_L.
    each value is computed from rows prior to the current row within the
    group (shift(1) before ewm), so the current row's value never leaks
    into its own feature.  missing values in source columns are ignored
    by the ewm so incomplete rows do not corrupt subsequent history.
    reusable for any sport: pass the appropriate group and date columns.
    """
    output: pd.DataFrame = df.sort_values(
        [group_col, date_col], kind="stable"
    ).copy()
    col: str
    for col in source_columns:
        if col not in output.columns:
            continue
        g = output.groupby(group_col, sort=False)[col]
        output[f"{col}_EWM_S"] = g.transform(
            lambda s: s.shift(1).ewm(
                span=short_span, min_periods=1, ignore_na=True
            ).mean()
        )
        output[f"{col}_EWM_L"] = g.transform(
            lambda s: s.shift(1).ewm(
                span=long_span, min_periods=1, ignore_na=True
            ).mean()
        )
    return output
