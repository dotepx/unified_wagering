from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from feature_building import (
    QuantileBucketSpec,
    add_calendar_categories,
    add_positive_quantile_target_buckets,
    make_column_token,
    make_log_ratio,
)
from market_data import (
    FEATURE_ADJ_CLOSE,
    FEATURE_CLOSE,
    FEATURE_HIGH,
    FEATURE_LOW,
    FEATURE_OPEN,
    FEATURE_VALUE,
    FEATURE_VOLUME,
    FRED_SERIES_COLUMNS,
    FRED_SERIES_ID_KEY,
    LABEL_DATE,
    LABEL_TARGET_DATE_PREFIX,
    MARKET_PRICE_COLUMNS,
    MARKET_PRICE_ID_KEY,
    SERIES_ID,
    SERIES_ID_ID,
    SERIES_SYMBOL,
    SERIES_SYMBOL_ID,
)
from utilities import (
    finalize_grammar_frame,
    load_parquet,
    make_file_stem,
    make_stable_uint_ids,
    make_save_to_parser,
    parallelize_df,
    resolve_identifiers,
    save_parquet,
    smart_type,
    uppercase_series_columns,
    update_status,
)


__all__: list[str] = []

if __name__ not in {"__main__", "__mp_main__"}:
    raise ImportError(
        "market_dataset_builder.py is a script entrypoint and should not be imported"
    )


@dataclass(frozen=True, slots=True)
class SeriesFrameContract:
    """hold one inbound series parquet contract"""

    frame_label: str
    value_columns: tuple[str, ...]
    series_column: str
    id_column: str
    id_key: tuple[str, ...]


PRICE_COLUMNS: list[str] = MARKET_PRICE_COLUMNS
PRICE_ID_KEY: list[str] = MARKET_PRICE_ID_KEY
FRED_COLUMNS: list[str] = FRED_SERIES_COLUMNS
FRED_ID_KEY: list[str] = FRED_SERIES_ID_KEY
AV_SERIES_CONTRACT: SeriesFrameContract = SeriesFrameContract(
    frame_label="av",
    value_columns=tuple(PRICE_COLUMNS),
    series_column=SERIES_SYMBOL,
    id_column=SERIES_SYMBOL_ID,
    id_key=tuple(PRICE_ID_KEY),
)
FRED_SERIES_CONTRACT: SeriesFrameContract = SeriesFrameContract(
    frame_label="fred",
    value_columns=tuple(FRED_COLUMNS),
    series_column=SERIES_ID,
    id_column=SERIES_ID_ID,
    id_key=tuple(FRED_ID_KEY),
)


@dataclass(frozen=True, slots=True)
class ResolvedAnchors:
    """hold resolved anchor ids"""

    av_symbols: tuple[str, ...]
    fred_series: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AppConfig:
    """hold runtime settings"""

    av_dir: Path
    fred_dir: Path
    save_to: Path
    symbols: tuple[str, ...]
    anchors: ResolvedAnchors
    forward_returns: tuple[int, ...]
    relative_symbol: str = ""
    threshold_series: str = ""
    threshold_series_scale: float = 0.0
    threshold_daily: float = 0.0
    positive_bucket_count: int = 9

    def __post_init__(self) -> None:
        if not self.av_dir.exists():
            raise ValueError(f"av_dir does not exist: {self.av_dir}")
        if not self.fred_dir.exists():
            raise ValueError(f"fred_dir does not exist: {self.fred_dir}")
        if not self.symbols:
            raise ValueError(
                "no av asset files resolved; place normalized av .pqt files in "
                "av_dir or provide --symbols/--symbols_csv for a smaller universe"
            )
        if not self.forward_returns:
            raise ValueError("at least one forward return period is required")
        if not all(period > 0 for period in self.forward_returns):
            raise ValueError(
                f"forward return periods must be positive: {self.forward_returns}"
            )
        if self.positive_bucket_count <= 0:
            raise ValueError(
                "positive_bucket_count must be positive; "
                f"received {self.positive_bucket_count}"
            )
        if self.save_to.name in {"", ".", ".."}:
            raise ValueError("save_to must include an output filename")
        if self.save_to.suffix.lower() not in {".pqt", ".parquet"}:
            raise ValueError("save_to must end with .pqt or .parquet")
        if self.threshold_series and self.threshold_series_scale == 0.0:
            raise ValueError(
                "threshold_series_scale must be non-zero when threshold_series is set"
            )


def parse_arguments() -> Namespace:
    """parse cli arguments"""
    parser: ArgumentParser = make_save_to_parser(
        description="build a normalized market feature and target dataset",
        save_to_help="output parquet path for the market dataset",
    )
    parser.add_argument(
        "--av_dir",
        type=Path,
        required=True,
        help="directory containing normalized av_loader .pqt files",
    )
    parser.add_argument(
        "--fred_dir",
        type=Path,
        required=True,
        help="directory containing normalized fred_loader .pqt files",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=[],
        help=(
            "optional smaller primary universe from cli, space or comma separated; "
            "defaults to every .pqt asset in av_dir"
        ),
    )
    parser.add_argument(
        "--symbols_csv",
        type=Path,
        default=None,
        help=(
            "optional csv whose first column contains a smaller primary universe; "
            "defaults to every .pqt asset in av_dir"
        ),
    )
    parser.add_argument(
        "--anchors",
        nargs="*",
        default=[],
        help="anchor ids from cli, resolved against av_dir or fred_dir",
    )
    parser.add_argument(
        "--anchors_csv",
        type=Path,
        default=None,
        help="csv whose first column contains anchor ids",
    )
    parser.add_argument(
        "--forward_returns",
        nargs="+",
        required=True,
        default=[],
        help="forward return periods in rows, space or comma separated",
    )
    parser.add_argument(
        "--relative_symbol",
        type=str,
        default="",
        help="optional av symbol whose forward return is subtracted from target",
    )
    parser.add_argument(
        "--threshold_series",
        type=str,
        default="",
        help="optional fred series whose current value contributes to hurdle",
    )
    parser.add_argument(
        "--threshold_series_scale",
        type=float,
        default=0.0,
        help="multiplier applied to threshold_series value per period row",
    )
    parser.add_argument(
        "--threshold_daily",
        type=float,
        default=0.0,
        help="constant per-row hurdle subtracted from each target",
    )
    parser.add_argument(
        "--positive_bucket_count",
        type=int,
        default=9,
        help="positive target buckets; one extra <=0 bucket is always added",
    )
    return parser.parse_args()


def scan_parquet_ids(directory: Path) -> dict[str, str]:
    """scan parquet ids by normalized stem"""
    return {
        path.stem.upper(): path.stem
        for path in sorted(directory.glob("*.pqt"))
        if path.is_file()
    }


def resolve_forward_returns(args: Namespace) -> tuple[int, ...]:
    """resolve forward return periods from args"""
    periods: list[int] = [
        int(token) for token in resolve_identifiers(args.forward_returns, None)
    ]
    return tuple(sorted(set(periods)))


def resolve_primary_symbols(args: Namespace) -> tuple[str, ...]:
    """resolve the primary universe from overrides or all av files"""
    symbols: tuple[str, ...] = resolve_identifiers(
        args.symbols,
        args.symbols_csv,
        uppercase=True,
    )
    if symbols:
        return symbols
    return tuple(scan_parquet_ids(args.av_dir).keys())


def resolve_anchors(args: Namespace) -> ResolvedAnchors:
    """resolve anchors into av symbols and fred series"""
    anchor_ids: tuple[str, ...] = resolve_identifiers(args.anchors, args.anchors_csv)
    if not anchor_ids:
        return ResolvedAnchors(av_symbols=(), fred_series=())

    av_ids: dict[str, str] = scan_parquet_ids(args.av_dir)
    fred_ids: dict[str, str] = scan_parquet_ids(args.fred_dir)
    av_symbols: list[str] = []
    fred_series: list[str] = []
    for anchor_id in anchor_ids:
        lookup_id: str = anchor_id.strip().upper()
        if lookup_id in av_ids and lookup_id in fred_ids:
            raise ValueError(
                f"anchor {anchor_id!r} is ambiguous; matching files exist in both "
                f"{args.av_dir} and {args.fred_dir}"
            )
        if lookup_id in av_ids:
            av_symbols.append(lookup_id)
            continue
        if lookup_id in fred_ids:
            fred_series.append(lookup_id)
            continue
        raise ValueError(
            f"anchor {anchor_id!r} did not resolve from {args.av_dir} or {args.fred_dir}"
        )
    return ResolvedAnchors(
        av_symbols=tuple(sorted(set(av_symbols))),
        fred_series=tuple(sorted(set(fred_series))),
    )


def resolve_optional_parquet_id(value: str, directory: Path, label: str) -> str:
    """resolve one optional id against a parquet directory"""
    item_id: str = value.strip()
    if not item_id:
        return ""
    lookup_id: str = item_id.upper()
    ids: dict[str, str] = scan_parquet_ids(directory)
    if lookup_id not in ids:
        raise ValueError(f"{label} {value!r} did not resolve from {directory}")
    return lookup_id


def build_app_config(args: Namespace) -> AppConfig:
    """build the app config"""
    return AppConfig(
        av_dir=args.av_dir,
        fred_dir=args.fred_dir,
        save_to=args.save_to,
        symbols=resolve_primary_symbols(args),
        anchors=resolve_anchors(args),
        forward_returns=resolve_forward_returns(args),
        relative_symbol=resolve_optional_parquet_id(
            args.relative_symbol,
            args.av_dir,
            "relative_symbol",
        ),
        threshold_series=resolve_optional_parquet_id(
            args.threshold_series,
            args.fred_dir,
            "threshold_series",
        ),
        threshold_series_scale=args.threshold_series_scale,
        threshold_daily=args.threshold_daily,
        positive_bucket_count=args.positive_bucket_count,
    )


def parquet_path(directory: Path, item_id: str) -> Path:
    """return the partitioned parquet path for one id"""
    candidate: Path = directory / f"{make_file_stem(item_id)}.pqt"
    if candidate.exists():
        return candidate
    matched_stem: str | None = scan_parquet_ids(directory).get(item_id.upper())
    return directory / f"{matched_stem}.pqt" if matched_stem else candidate


def require_columns(df: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    """fail fast when required columns are missing"""
    missing: list[str] = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {missing}")


def validate_stable_id_values(
    df: pd.DataFrame,
    source_column: str,
    id_column: str,
) -> None:
    """validate stable uint ids for one series column"""
    require_columns(df, [source_column, id_column], "series id validation")
    cleaned_values: pd.Series = df[source_column].astype("string").str.strip()
    if cleaned_values.isna().any() or cleaned_values.eq("").any():
        raise ValueError(f"id source column contains empty values: {source_column}")
    expected_ids: pd.Series = make_stable_uint_ids(cleaned_values)
    actual_ids: pd.Series = pd.to_numeric(df[id_column], errors="raise").astype(
        "uint64"
    )
    if not actual_ids.reset_index(drop=True).equals(
        expected_ids.reset_index(drop=True)
    ):
        raise ValueError(f"{id_column} does not match stable ids for {source_column}")


def load_series_frame(
    directory: Path,
    item_id: str,
    contract: SeriesFrameContract,
) -> pd.DataFrame:
    """load one normalized series parquet file"""
    path: Path = parquet_path(directory, item_id)
    df: pd.DataFrame = load_parquet(path)
    require_columns(
        df,
        [*contract.value_columns, contract.id_column],
        f"{item_id} {contract.frame_label} file",
    )
    columns: list[str] = [*contract.value_columns, contract.id_column]
    output: pd.DataFrame = df.loc[:, columns].copy()
    output = uppercase_series_columns(output)
    output[LABEL_DATE] = pd.to_datetime(output[LABEL_DATE], errors="raise")
    numeric_columns: list[str] = [
        column
        for column in contract.value_columns
        if column not in {contract.series_column, LABEL_DATE}
    ]
    output = output.astype({column: "float64" for column in numeric_columns})
    output[contract.id_column] = output[contract.id_column].astype("uint64")
    validate_stable_id_values(output, contract.series_column, contract.id_column)
    if output.duplicated(list(contract.id_key)).any():
        raise ValueError(
            f"{item_id} {contract.frame_label} file contains duplicate date rows"
        )
    return output.sort_values(
        list(contract.id_key),
        kind="stable",
        ignore_index=True,
    )


def load_av_frame(symbol: str, av_dir: Path) -> pd.DataFrame:
    """load one normalized av symbol file"""
    return load_series_frame(
        av_dir,
        symbol,
        AV_SERIES_CONTRACT,
    )


def load_fred_frame(series_id: str, fred_dir: Path) -> pd.DataFrame:
    """load one normalized fred series file"""
    return load_series_frame(
        fred_dir,
        series_id,
        FRED_SERIES_CONTRACT,
    )


def load_av_frames(symbols: set[str], av_dir: Path) -> pd.DataFrame:
    """load many av symbol files"""
    return parallelize_df(
        partial(load_av_frame, av_dir=av_dir),
        list(symbols),
    )


def load_fred_frames(series_ids: set[str], fred_dir: Path) -> pd.DataFrame:
    """load many fred series files"""
    if not series_ids:
        return pd.DataFrame(columns=FRED_COLUMNS + [SERIES_ID_ID])
    return parallelize_df(
        partial(load_fred_frame, fred_dir=fred_dir),
        list(series_ids),
    )


def filter_series_frame(
    df: pd.DataFrame, series_column: str, series_ids: set[str]
) -> pd.DataFrame:
    """filter a concatenated series frame to selected ids"""
    if not series_ids:
        return df.iloc[0:0].copy()
    return df.loc[df[series_column].isin(series_ids)].copy()


def build_primary_panel(
    av_frame: pd.DataFrame, symbols: tuple[str, ...]
) -> pd.DataFrame:
    """build the primary market panel"""
    output: pd.DataFrame = filter_series_frame(
        av_frame,
        SERIES_SYMBOL,
        set(symbols),
    )
    return output.sort_values(PRICE_ID_KEY, kind="stable", ignore_index=True)


def add_market_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """add per-symbol market features"""
    output: pd.DataFrame = df.sort_values(
        PRICE_ID_KEY,
        kind="stable",
        ignore_index=True,
    ).copy()
    grouped = output.groupby(SERIES_SYMBOL_ID, sort=False)
    prev_close: pd.Series = grouped[FEATURE_CLOSE].shift(1)
    prev_adj_close: pd.Series = grouped[FEATURE_ADJ_CLOSE].shift(1)
    prev_volume: pd.Series = grouped[FEATURE_VOLUME].shift(1)
    dollar_volume: pd.Series = output[FEATURE_VOLUME] * output[FEATURE_ADJ_CLOSE]
    prev_dollar_volume: pd.Series = dollar_volume.groupby(
        output[SERIES_SYMBOL_ID],
        sort=False,
    ).shift(1)

    output["FEATURE_LOG_ADJ_CLOSE_TO_CLOSE_RETURN"] = make_log_ratio(
        output[FEATURE_ADJ_CLOSE],
        output[FEATURE_CLOSE],
    )
    output["FEATURE_LOG_OPEN_TO_CLOSE_RETURN"] = make_log_ratio(
        output[FEATURE_CLOSE],
        output[FEATURE_OPEN],
    )
    output["FEATURE_LOG_RANGE"] = make_log_ratio(
        output[FEATURE_HIGH],
        output[FEATURE_LOW],
    )
    output["FEATURE_LOG_ADJ_DOLLAR_VOLUME_RETURN"] = make_log_ratio(
        dollar_volume,
        prev_dollar_volume,
    )
    output["FEATURE_LOG_CLOSE_RETURN"] = make_log_ratio(
        output[FEATURE_CLOSE],
        prev_close,
    )
    output["FEATURE_LOG_ADJ_CLOSE_RETURN"] = make_log_ratio(
        output[FEATURE_ADJ_CLOSE],
        prev_adj_close,
    )
    output["FEATURE_LOG_OPEN_GAP_RETURN"] = make_log_ratio(
        output[FEATURE_OPEN],
        prev_close,
    )
    output["FEATURE_LOG_VOLUME_RETURN"] = make_log_ratio(
        output[FEATURE_VOLUME],
        prev_volume,
    )
    output["FEATURE_LOG_ADJ_DOLLAR_VOLUME"] = pd.Series(
        np.nan,
        index=output.index,
        dtype="float64",
    )
    valid_dollar_volume: pd.Series = dollar_volume.gt(0)
    output.loc[valid_dollar_volume, "FEATURE_LOG_ADJ_DOLLAR_VOLUME"] = np.log(
        dollar_volume.loc[valid_dollar_volume]
    )
    return output


def build_anchor_return_frame(
    av_frame: pd.DataFrame,
    anchor_symbols: tuple[str, ...],
) -> pd.DataFrame:
    """build a wide frame of av anchor returns by date"""
    if not anchor_symbols:
        return pd.DataFrame(columns=[LABEL_DATE])

    frames: list[pd.DataFrame] = []
    for symbol in anchor_symbols:
        frame: pd.DataFrame = filter_series_frame(
            av_frame,
            SERIES_SYMBOL,
            {symbol},
        ).sort_values(
            PRICE_ID_KEY,
            kind="stable",
            ignore_index=True,
        )
        log_return: pd.Series = make_log_ratio(
            frame[FEATURE_ADJ_CLOSE],
            frame[FEATURE_ADJ_CLOSE].shift(1),
        )
        feature_name: str = f"FEATURE_ANCHOR_{make_column_token(symbol)}_LOG_RETURN"
        frames.append(
            pd.DataFrame(
                {
                    LABEL_DATE: frame[LABEL_DATE],
                    feature_name: log_return,
                    f"{feature_name}_SQUARED": log_return.pow(2),
                }
            ).drop_duplicates(LABEL_DATE)
        )
    return merge_date_frames(frames)


def build_fred_return_frame(
    fred_frame: pd.DataFrame,
    feature_series: tuple[str, ...],
) -> pd.DataFrame:
    """build a wide frame of fred return features by date"""
    if not feature_series:
        return pd.DataFrame(columns=[LABEL_DATE])

    frames: list[pd.DataFrame] = []
    for series_id in feature_series:
        frame: pd.DataFrame = filter_series_frame(
            fred_frame,
            SERIES_ID,
            {series_id},
        ).sort_values(
            FRED_ID_KEY,
            kind="stable",
            ignore_index=True,
        )
        log_return: pd.Series = make_log_ratio(
            frame[FEATURE_VALUE],
            frame[FEATURE_VALUE].shift(1),
        )
        feature_name: str = f"FEATURE_FRED_{make_column_token(series_id)}_LOG_RETURN"
        frames.append(
            pd.DataFrame(
                {
                    LABEL_DATE: frame[LABEL_DATE],
                    feature_name: log_return,
                    f"{feature_name}_SQUARED": log_return.pow(2),
                }
            ).drop_duplicates(LABEL_DATE)
        )
    return merge_date_frames(frames)


def merge_date_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """merge date-keyed feature frames"""
    if not frames:
        return pd.DataFrame(columns=[LABEL_DATE])
    output: pd.DataFrame = frames[0]
    for frame in frames[1:]:
        output = output.merge(frame, on=LABEL_DATE, how="outer", sort=True)
    return output.sort_values(LABEL_DATE, kind="stable", ignore_index=True)


def add_external_features(
    df: pd.DataFrame,
    av_frame: pd.DataFrame,
    fred_frame: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    """join anchor return features onto the panel"""
    output: pd.DataFrame = df.copy()
    anchor_features: pd.DataFrame = build_anchor_return_frame(
        av_frame,
        config.anchors.av_symbols,
    )
    if len(anchor_features.columns) > 1:
        output = output.merge(anchor_features, on=LABEL_DATE, how="left", sort=False)

    fred_features: pd.DataFrame = build_fred_return_frame(
        fred_frame,
        config.anchors.fred_series,
    )
    if len(fred_features.columns) > 1:
        output = output.merge(fred_features, on=LABEL_DATE, how="left", sort=False)
    return output


def build_symbol_baseline_frame(
    av_frame: pd.DataFrame,
    relative_symbol: str,
    forward_returns: tuple[int, ...],
) -> pd.DataFrame:
    """build baseline forward returns from one av symbol"""
    if not relative_symbol:
        return pd.DataFrame(columns=[LABEL_DATE])

    frame: pd.DataFrame = filter_series_frame(
        av_frame,
        SERIES_SYMBOL,
        {relative_symbol},
    ).sort_values(
        PRICE_ID_KEY,
        kind="stable",
        ignore_index=True,
    )
    output: pd.DataFrame = pd.DataFrame({LABEL_DATE: frame[LABEL_DATE]})
    for period in forward_returns:
        output[f"_BASELINE_SYMBOL_{period}"] = make_log_ratio(
            frame[FEATURE_ADJ_CLOSE].shift(-period),
            frame[FEATURE_ADJ_CLOSE],
        )
    return output.drop_duplicates(LABEL_DATE)


def build_threshold_series_frame(
    fred_frame: pd.DataFrame,
    threshold_series: str,
    threshold_series_scale: float,
    forward_returns: tuple[int, ...],
) -> pd.DataFrame:
    """build horizon-scaled threshold values from one fred series"""
    if not threshold_series:
        return pd.DataFrame(columns=[LABEL_DATE])

    frame: pd.DataFrame = filter_series_frame(
        fred_frame,
        SERIES_ID,
        {threshold_series},
    ).sort_values(
        FRED_ID_KEY,
        kind="stable",
        ignore_index=True,
    )
    output: pd.DataFrame = pd.DataFrame({LABEL_DATE: frame[LABEL_DATE]})
    for period in forward_returns:
        output[f"_THRESHOLD_SERIES_{period}"] = (
            frame[FEATURE_VALUE].astype("float64") * threshold_series_scale * period
        )
    return output.drop_duplicates(LABEL_DATE)


def add_forward_targets(
    df: pd.DataFrame,
    av_frame: pd.DataFrame,
    fred_frame: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    """add forward market target columns"""
    output: pd.DataFrame = df.sort_values(
        PRICE_ID_KEY,
        kind="stable",
        ignore_index=True,
    ).copy()
    symbol_baseline: pd.DataFrame = build_symbol_baseline_frame(
        av_frame,
        config.relative_symbol,
        config.forward_returns,
    )
    if len(symbol_baseline.columns) > 1:
        output = output.merge(symbol_baseline, on=LABEL_DATE, how="left", sort=False)

    series_baseline: pd.DataFrame = build_threshold_series_frame(
        fred_frame,
        config.threshold_series,
        config.threshold_series_scale,
        config.forward_returns,
    )
    if len(series_baseline.columns) > 1:
        output = output.merge(series_baseline, on=LABEL_DATE, how="left", sort=False)

    grouped = output.groupby(SERIES_SYMBOL_ID, sort=False)
    for period in config.forward_returns:
        output[f"{LABEL_TARGET_DATE_PREFIX}_{period}"] = grouped[LABEL_DATE].shift(
            -period
        )
        output[f"TARGET_RAW_LOG_RETURN_{period}"] = make_log_ratio(
            grouped[FEATURE_ADJ_CLOSE].shift(-period),
            output[FEATURE_ADJ_CLOSE],
        )
        baseline: pd.Series = pd.Series(0.0, index=output.index, dtype="float64")
        symbol_col: str = f"_BASELINE_SYMBOL_{period}"
        if symbol_col in output.columns:
            baseline = baseline.add(output[symbol_col].fillna(0.0), fill_value=0.0)
        series_col: str = f"_THRESHOLD_SERIES_{period}"
        if series_col in output.columns:
            baseline = baseline.add(output[series_col].fillna(0.0), fill_value=0.0)
        if config.threshold_daily != 0.0:
            baseline = baseline.add(config.threshold_daily * period, fill_value=0.0)
        output[f"TARGET_BASELINE_LOG_RETURN_{period}"] = baseline
        output[f"TARGET_EXCESS_LOG_RETURN_{period}"] = (
            output[f"TARGET_RAW_LOG_RETURN_{period}"] - baseline
        )

    drop_cols: list[str] = [
        column for column in output.columns if column.startswith("_")
    ]
    return output.drop(columns=drop_cols)


def add_bucket_targets(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """add reusable quantile bucket targets"""
    target_columns: list[str] = [
        f"TARGET_EXCESS_LOG_RETURN_{period}" for period in config.forward_returns
    ]
    return add_positive_quantile_target_buckets(
        df,
        target_columns,
        QuantileBucketSpec(config.positive_bucket_count),
    )


def validate_market_dataset(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """validate the market builder output contract"""
    require_columns(df, PRICE_COLUMNS + [SERIES_SYMBOL_ID], "market dataset")
    required_targets: list[str] = [
        f"TARGET_EXCESS_LOG_RETURN_{period}" for period in config.forward_returns
    ]
    require_columns(df, required_targets, "market dataset")
    if df.duplicated(PRICE_ID_KEY).any():
        raise ValueError("market dataset contains duplicate symbol/date rows")
    return df


def save_output(df: pd.DataFrame, config: AppConfig) -> None:
    """save the market dataset"""
    save_parquet(df, config.save_to)
    update_status(f"saved market dataset to {config.save_to}")


def main() -> None:
    """run the market dataset builder"""
    config: AppConfig = build_app_config(parse_arguments())
    required_av_ids: set[str] = set(config.symbols) | set(config.anchors.av_symbols)
    if config.relative_symbol:
        required_av_ids.add(config.relative_symbol)
    required_fred_ids: set[str] = set(config.anchors.fred_series)
    if config.threshold_series:
        required_fred_ids.add(config.threshold_series)

    update_status(
        f"loading {len(config.symbols)} primary symbols, "
        f"{len(config.anchors.av_symbols)} av anchors, "
        f"{len(config.anchors.fred_series)} fred anchors"
    )
    av_frame: pd.DataFrame = load_av_frames(required_av_ids, config.av_dir)
    fred_frame: pd.DataFrame = load_fred_frames(required_fred_ids, config.fred_dir)

    output: pd.DataFrame = build_primary_panel(av_frame, config.symbols)
    output = add_calendar_categories(output)
    output = add_market_price_features(output)
    output = add_external_features(output, av_frame, fred_frame, config)
    output = add_forward_targets(output, av_frame, fred_frame, config)
    output = add_bucket_targets(output, config)
    output = validate_market_dataset(output, config)
    output = smart_type(output)
    output = finalize_grammar_frame(output)
    save_output(output, config)


if __name__ == "__main__":
    main()
