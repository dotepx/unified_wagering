from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandera.pandas as pa
import pandas as pd
from pandera import Check
import requests

from market_data import (
    FEATURE_VALUE,
    FRED_SERIES_COLUMNS,
    FRED_SERIES_KEY,
    LABEL_DATE,
    SERIES_ID,
)
from utilities import (
    BaseAppConfig,
    DownloadResult,
    FetchedFrame,
    add_api_key_args,
    add_series_id_columns,
    download_with_passes,
    finalize_grammar_frame,
    make_save_to_parser,
    resolve_api_key,
    resolve_identifiers,
    save_csv,
    save_keyed_parquet_files,
    update_status,
    validate_frame,
)


__all__: list[str] = []

if __name__ not in {"__main__", "__mp_main__"}:
    raise ImportError(
        "fred_loader.py is a script entrypoint and should not be imported"
    )


FRED_API_KEY_ENV: Final[str] = "FRED_API_KEY"
BASE_SERIES_COLUMNS: Final[list[str]] = FRED_SERIES_COLUMNS
SERIES_DATE_KEY: Final[list[str]] = FRED_SERIES_KEY


@dataclass(frozen=True, slots=True)
class AppConfig(BaseAppConfig):
    """hold runtime settings"""

    api_key: str
    download_item_ids: tuple[str, ...]
    item_label: str = "series"
    failed_ids_file: str = "failed_series_ids.csv"
    download_passes: int = 2
    request_spacing: float = 0.25
    base_url: str = "https://api.stlouisfed.org/fred/series/observations"
    timeout: tuple[float, float] = (5.0, 5.0)
    observation_start: str = ""
    observation_end: str = ""

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError(
                "missing fred api key; use --api_key, --api_key_file, "
                f"or set {FRED_API_KEY_ENV}"
            )
        if not self.download_item_ids:
            raise ValueError(
                f"no {self.item_label} ids provided via --series_ids and/or --series_csv"
            )
        if self.download_passes <= 0:
            raise ValueError(
                f"download_passes must be positive; received {self.download_passes}"
            )
        if self.request_spacing <= 0.0:
            raise ValueError(
                f"request_spacing must be positive; received {self.request_spacing}"
            )
        if not all(timeout > 0 for timeout in self.timeout):
            raise ValueError(
                f"timeout values must be positive; received {self.timeout}"
            )
        if self.observation_start:
            pd.to_datetime(
                self.observation_start,
                format="%Y-%m-%d",
                errors="raise",
            )
        if self.observation_end:
            pd.to_datetime(
                self.observation_end,
                format="%Y-%m-%d",
                errors="raise",
            )
        if self.observation_start and self.observation_end:
            if self.observation_start > self.observation_end:
                raise ValueError(
                    "observation_start must be on or before observation_end"
                )
        BaseAppConfig.__post_init__(self)

    @property
    def failed_ids_path(self) -> Path:
        """return failed ids output path"""
        return self.output_dir / self.failed_ids_file


def parse_arguments() -> Namespace:
    """parse cli arguments"""
    parser: ArgumentParser = make_save_to_parser(
        "download fred observations for one or more series ids",
        save_to_help="output directory",
    )
    add_api_key_args(parser, "fred")
    parser.add_argument(
        "--series_ids",
        nargs="*",
        default=[],
        help="fred series ids from cli, space or comma separated",
    )
    parser.add_argument(
        "--series_csv",
        type=Path,
        default=None,
        help="csv whose first column contains fred series ids",
    )
    parser.add_argument(
        "--download_passes",
        type=int,
        default=2,
        help="number of passes over unresolved series ids",
    )
    parser.add_argument(
        "--request_spacing",
        type=float,
        default=0.25,
        help="seconds between fred request starts",
    )
    parser.add_argument(
        "--observation_start",
        type=str,
        default="",
        help="optional observation start date in yyyy-mm-dd format",
    )
    parser.add_argument(
        "--observation_end",
        type=str,
        default="",
        help="optional observation end date in yyyy-mm-dd format",
    )
    return parser.parse_args()


def resolve_series_ids(args: Namespace) -> tuple[str, ...]:
    """resolve series ids from args"""
    return resolve_identifiers(args.series_ids, args.series_csv, uppercase=True)


def build_app_config(args: Namespace) -> AppConfig:
    """build the app config"""
    return AppConfig(
        api_key=resolve_api_key(
            args.api_key,
            args.api_key_file,
            FRED_API_KEY_ENV,
            "fred",
        ),
        save_to=args.save_to,
        download_item_ids=resolve_series_ids(args),
        download_passes=args.download_passes,
        request_spacing=args.request_spacing,
        observation_start=args.observation_start,
        observation_end=args.observation_end,
    )


def base_series_columns() -> dict[str, pa.Column]:
    """build the base series columns"""
    return {
        SERIES_ID: pa.Column(
            str,
            nullable=False,
            checks=Check.str_matches(r"^[A-Za-z0-9._\-]+$"),
        ),
        LABEL_DATE: pa.Column(pa.DateTime, nullable=False),
        FEATURE_VALUE: pa.Column(
            float,
            nullable=False,
            checks=Check(
                lambda series: np.isfinite(series.to_numpy()).all(),
                error="values must be finite",
            ),
        ),
    }


def frame_has_unique_series_dates(df: pd.DataFrame) -> bool:
    """check series date uniqueness"""
    return not df.duplicated(SERIES_DATE_KEY).any()


def frame_is_sorted_by_series_date(df: pd.DataFrame) -> bool:
    """check series date ordering"""
    current: pd.DataFrame = df.loc[:, SERIES_DATE_KEY].reset_index(drop=True)
    expected: pd.DataFrame = (
        df.sort_values(SERIES_DATE_KEY, kind="stable")
        .loc[:, SERIES_DATE_KEY]
        .reset_index(drop=True)
    )
    return current.equals(expected)


def series_frame_checks() -> list[Check]:
    """build the series frame checks"""
    return [
        Check(
            frame_has_unique_series_dates,
            error="duplicate series_id and tradedate rows",
        ),
        Check(
            frame_is_sorted_by_series_date,
            error="rows must be sorted by series_id and tradedate",
        ),
    ]


def series_schema() -> pa.DataFrameSchema:
    """build the fred series schema"""
    return pa.DataFrameSchema(
        columns=base_series_columns(),
        checks=series_frame_checks(),
        strict=True,
        ordered=True,
        coerce=True,
    )


def parse_series_frame(
    observations: Sequence[Mapping[str, Any]],
    series_id: str,
) -> pd.DataFrame:
    """parse one fred series payload"""
    frame: pd.DataFrame = pd.DataFrame(observations)
    missing_columns: list[str] = [
        column for column in ("date", "value") if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"missing expected columns {missing_columns}; available={sorted(frame.columns)}"
        )
    output: pd.DataFrame = pd.DataFrame(
        {
            LABEL_DATE: pd.to_datetime(
                frame["date"],
                format="%Y-%m-%d",
                errors="raise",
            ),
            FEATURE_VALUE: pd.to_numeric(frame["value"], errors="coerce"),
        }
    ).dropna(subset=[FEATURE_VALUE])
    if output.empty:
        raise ValueError(
            "no numeric observations returned after dropping missing values"
        )
    output.insert(0, SERIES_ID, series_id.upper())
    return output.loc[:, BASE_SERIES_COLUMNS].sort_values(
        SERIES_DATE_KEY,
        kind="stable",
        ignore_index=True,
    )


def get_series_observations(series_id: str, config: AppConfig) -> pd.DataFrame:
    """download one fred series history"""
    params: dict[str, str] = {
        "series_id": series_id,
        "api_key": config.api_key,
        "file_type": "json",
        "units": "lin",
    }
    if config.observation_start:
        params["observation_start"] = config.observation_start
    if config.observation_end:
        params["observation_end"] = config.observation_end
    response: requests.Response = requests.get(
        config.base_url,
        params=params,
        timeout=config.timeout,
    )
    response.raise_for_status()
    payload: Any = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a json object, received {type(payload)}")
    message: str = next(
        (
            str(payload[key])
            for key in ("error_message", "error", "message")
            if payload.get(key)
        ),
        "",
    )
    if message:
        raise ValueError(message)
    observations: Any = payload.get("observations")
    if not isinstance(observations, Sequence) or not observations:
        raise ValueError(f"missing {'observations'!r}; keys={sorted(payload.keys())}")
    return validate_frame(
        parse_series_frame(observations, series_id),
        series_schema(),
        f"{series_id} raw series",
    )


def fetch_series_observations(series_id: str, config: AppConfig) -> FetchedFrame:
    """fetch one fred series safely"""
    try:
        return FetchedFrame(
            item_id=series_id,
            frame=get_series_observations(series_id, config),
        )
    except (requests.RequestException, ValueError) as ex:
        update_status(f"{series_id}: {ex}")
        return FetchedFrame(item_id=series_id, frame=pd.DataFrame())


def empty_series_frame() -> pd.DataFrame:
    """return an empty fred series frame with the expected columns"""
    return pd.DataFrame(columns=BASE_SERIES_COLUMNS)


def download_series(config: AppConfig) -> DownloadResult:
    """download all requested fred series"""
    download: DownloadResult = download_with_passes(
        config,
        fetch_one=fetch_series_observations,
        fallback_frame=empty_series_frame(),
    )
    dataset: pd.DataFrame = download.dataset.sort_values(
        SERIES_DATE_KEY,
        kind="stable",
        ignore_index=True,
    )
    if not dataset.empty:
        dataset = validate_frame(dataset, series_schema(), "fred series")
    return DownloadResult(dataset=dataset, failed_ids=download.failed_ids)


def save_outputs(
    series_df: pd.DataFrame, failed_ids: Sequence[str], config: AppConfig
) -> None:
    """save the output files"""
    if series_df.empty:
        raise RuntimeError(
            f"no fred data returned for series_ids={config.download_item_ids}; "
            f"failed_ids={failed_ids}"
        )
    series_df = add_series_id_columns(series_df)
    series_df = finalize_grammar_frame(series_df)
    saved_count: int = save_keyed_parquet_files(series_df, SERIES_ID, config.output_dir)
    update_status(f"saved {saved_count} series files to {config.output_dir}")

    if failed_ids:
        save_csv(
            pd.DataFrame({SERIES_ID: failed_ids}),
            config.failed_ids_path,
        )
        update_status(f"saved {len(failed_ids)} failed ids to {config.failed_ids_path}")


def main() -> None:
    """run the fred loader"""
    config: AppConfig = build_app_config(parse_arguments())

    update_status(f"resolved {len(config.download_item_ids)} series ids")
    download: DownloadResult = download_series(config)

    save_outputs(download.dataset, download.failed_ids, config)


if __name__ == "__main__":
    main()
