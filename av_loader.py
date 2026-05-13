from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pandera.pandas as pa
import requests
from pandera import Check

from market_data import (
    FEATURE_ADJ_CLOSE,
    FEATURE_CLOSE,
    FEATURE_HIGH,
    FEATURE_LOW,
    FEATURE_OPEN,
    FEATURE_VOLUME,
    LABEL_DATE,
    MARKET_PRICE_COLUMNS,
    MARKET_PRICE_KEY,
    MARKET_PRICE_VALUE_COLUMNS,
    SERIES_SYMBOL,
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
    raise ImportError("av_loader.py is a script entrypoint and should not be imported")


ALPHA_VANTAGE_API_KEY_ENV: Final[str] = "ALPHA_VANTAGE_API_KEY"
BASE_PRICE_COLUMNS: Final[list[str]] = MARKET_PRICE_COLUMNS
SYMBOL_DATE_KEY: Final[list[str]] = MARKET_PRICE_KEY
PRICE_VALUE_COLUMNS: Final[list[str]] = MARKET_PRICE_VALUE_COLUMNS


@dataclass(frozen=True, slots=True)
class AppConfig(BaseAppConfig):
    """hold runtime settings"""

    api_key: str
    download_item_ids: tuple[str, ...]
    item_label: str = "symbol"
    failed_symbols_file: str = "failed_symbols.csv"
    download_passes: int = 2
    request_spacing: float = 0.25
    base_url: str = "https://www.alphavantage.co/query"
    function_name: str = "TIME_SERIES_DAILY_ADJUSTED"
    timeout: tuple[float, float] = (5.0, 5.0)

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError(
                "missing alpha vantage api key; use --api_key, --api_key_file, "
                f"or set {ALPHA_VANTAGE_API_KEY_ENV}"
            )
        if not self.download_item_ids:
            raise ValueError(
                f"no {self.item_label}s provided via --symbols and/or --symbols_csv"
            )
        if self.download_passes <= 0:
            raise ValueError(
                f"download_passes must be positive; received {self.download_passes}"
            )
        if self.request_spacing <= 0.0:
            raise ValueError(
                f"request_spacing must be positive; received {self.request_spacing}"
            )
        if self.request_spacing < 0.2:
            raise ValueError(
                "request_spacing must be at least 0.2 seconds to respect the "
                "alpha vantage 5 requests per second burst gate"
            )
        if not all(timeout > 0 for timeout in self.timeout):
            raise ValueError(
                f"timeout values must be positive; received {self.timeout}"
            )
        BaseAppConfig.__post_init__(self)

    @property
    def failed_symbols_path(self) -> Path:
        """return failed symbols output path"""
        return self.output_dir / self.failed_symbols_file


def parse_arguments() -> Namespace:
    """parse cli arguments"""
    parser: ArgumentParser = make_save_to_parser(
        "download raw alpha vantage daily adjusted prices",
        save_to_help="output directory",
    )
    add_api_key_args(parser, "alpha vantage")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=[],
        help="symbols from cli, space or comma separated",
    )
    parser.add_argument(
        "--symbols_csv",
        type=Path,
        default=None,
        help="csv whose first column contains ticker symbols",
    )
    parser.add_argument(
        "--download_passes",
        type=int,
        default=2,
        help="number of passes over unresolved symbols",
    )
    parser.add_argument(
        "--request_spacing",
        type=float,
        default=0.25,
        help="seconds between alpha vantage request starts",
    )
    return parser.parse_args()


def resolve_symbols(args: Namespace) -> tuple[str, ...]:
    """resolve symbols from args"""
    return resolve_identifiers(args.symbols, args.symbols_csv, uppercase=True)


def build_app_config(args: Namespace) -> AppConfig:
    """build the app config"""
    return AppConfig(
        api_key=resolve_api_key(
            args.api_key,
            args.api_key_file,
            ALPHA_VANTAGE_API_KEY_ENV,
            "alpha vantage",
        ),
        save_to=args.save_to,
        download_item_ids=resolve_symbols(args),
        download_passes=args.download_passes,
        request_spacing=args.request_spacing,
    )


def base_price_columns() -> dict[str, pa.Column]:
    """build the base price columns"""
    return {
        SERIES_SYMBOL: pa.Column(
            str,
            nullable=False,
            checks=Check.str_matches(r"^[A-Z0-9.\-^/]+$"),
        ),
        LABEL_DATE: pa.Column(pa.DateTime, nullable=False),
        FEATURE_OPEN: pa.Column(float, nullable=False, checks=Check.gt(0.0)),
        FEATURE_HIGH: pa.Column(float, nullable=False, checks=Check.gt(0.0)),
        FEATURE_LOW: pa.Column(float, nullable=False, checks=Check.gt(0.0)),
        FEATURE_CLOSE: pa.Column(float, nullable=False, checks=Check.gt(0.0)),
        FEATURE_ADJ_CLOSE: pa.Column(float, nullable=False, checks=Check.gt(0.0)),
        FEATURE_VOLUME: pa.Column(float, nullable=False, checks=Check.ge(0.0)),
    }


def frame_has_unique_symbol_dates(df: pd.DataFrame) -> bool:
    """check symbol date uniqueness"""
    return not df.duplicated(SYMBOL_DATE_KEY).any()


def frame_is_sorted_by_symbol_date(df: pd.DataFrame) -> bool:
    """check symbol date ordering"""
    current: pd.DataFrame = df.loc[:, SYMBOL_DATE_KEY].reset_index(drop=True)
    expected: pd.DataFrame = (
        df.sort_values(SYMBOL_DATE_KEY, kind="stable")
        .loc[:, SYMBOL_DATE_KEY]
        .reset_index(drop=True)
    )
    return current.equals(expected)


def frame_has_valid_ohlc_bounds(df: pd.DataFrame) -> bool:
    """check ohlc bounds"""
    high_ok: pd.Series = df[FEATURE_HIGH].ge(
        df.loc[:, [FEATURE_OPEN, FEATURE_CLOSE, FEATURE_LOW]].max(axis=1)
    )
    low_ok: pd.Series = df[FEATURE_LOW].le(
        df.loc[:, [FEATURE_OPEN, FEATURE_CLOSE, FEATURE_HIGH]].min(axis=1)
    )
    return bool((high_ok & low_ok).all())


def raw_frame_checks() -> list[Check]:
    """build the raw frame checks"""
    return [
        Check(
            frame_has_unique_symbol_dates,
            error="duplicate symbol and tradedate rows",
        ),
        Check(
            frame_is_sorted_by_symbol_date,
            error="rows must be sorted by symbol and tradedate",
        ),
        Check(
            frame_has_valid_ohlc_bounds,
            error="ohlc values must stay within low and high bounds",
        ),
    ]


def raw_prices_schema() -> pa.DataFrameSchema:
    """build the raw prices schema"""
    return pa.DataFrameSchema(
        columns=base_price_columns(),
        checks=raw_frame_checks(),
        strict=True,
        ordered=True,
        coerce=True,
    )


def empty_price_frame() -> pd.DataFrame:
    """return an empty raw price frame with the expected columns"""
    return pd.DataFrame(columns=BASE_PRICE_COLUMNS)


def parse_price_frame(
    time_series: Mapping[str, Mapping[str, Any]],
    symbol: str,
) -> pd.DataFrame:
    """parse one price payload"""
    raw_to_price_columns: dict[str, str] = {
        "1. open": FEATURE_OPEN,
        "2. high": FEATURE_HIGH,
        "3. low": FEATURE_LOW,
        "4. close": FEATURE_CLOSE,
        "5. adjusted close": FEATURE_ADJ_CLOSE,
        "6. volume": FEATURE_VOLUME,
    }
    frame: pd.DataFrame = pd.DataFrame.from_dict(dict(time_series), orient="index")
    missing_columns: list[str] = [
        column for column in raw_to_price_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"missing expected columns {missing_columns}; available={sorted(frame.columns)}"
        )
    prices: pd.DataFrame = (
        frame.rename(columns=raw_to_price_columns)
        .loc[:, PRICE_VALUE_COLUMNS]
        .rename_axis(LABEL_DATE)
        .reset_index()
        .astype({column: "float64" for column in PRICE_VALUE_COLUMNS})
    )
    prices[LABEL_DATE] = pd.to_datetime(
        prices[LABEL_DATE],
        format="%Y-%m-%d",
        errors="raise",
    )
    prices.insert(0, SERIES_SYMBOL, symbol.upper())
    return prices.loc[:, BASE_PRICE_COLUMNS].sort_values(
        SYMBOL_DATE_KEY,
        kind="stable",
        ignore_index=True,
    )


def get_prices(symbol: str, config: AppConfig) -> pd.DataFrame:
    """download one symbol price history"""
    response: requests.Response = requests.get(
        config.base_url,
        params={
            "function": config.function_name,
            "symbol": symbol,
            "outputsize": "full",
            "apikey": config.api_key,
        },
        timeout=config.timeout,
    )
    response.raise_for_status()
    payload: Any = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a json object, received {type(payload)}")
    message: str = next(
        (
            str(payload[key])
            for key in ("Error Message", "Note", "Information")
            if payload.get(key)
        ),
        "",
    )
    if message:
        raise ValueError(message)
    time_series: Any = payload.get("Time Series (Daily)")
    if not isinstance(time_series, Mapping) or not time_series:
        raise ValueError(
            f"missing {'Time Series (Daily)'!r}; keys={sorted(payload.keys())}"
        )
    return validate_frame(
        parse_price_frame(time_series, symbol),
        raw_prices_schema(),
        f"{symbol} raw prices",
    )


def is_minute_limit_message(message: str) -> bool:
    """check for alpha vantage minute-limit notes"""
    lowered: str = message.lower()
    return (
        "minute-level rate limit exceed" in lowered
        or "api requests per minute" in lowered
        or "premium subscription plan" in lowered
    )


def fetch_symbol_prices(symbol: str, config: AppConfig) -> FetchedFrame:
    """fetch one symbol safely"""
    try:
        return FetchedFrame(item_id=symbol, frame=get_prices(symbol, config))
    except (requests.RequestException, ValueError) as ex:
        update_status(f"{symbol}: {ex}")
        return FetchedFrame(
            item_id=symbol,
            frame=empty_price_frame(),
            cooldown_seconds=15.0 if is_minute_limit_message(str(ex)) else 0.0,
        )


def download_prices(config: AppConfig) -> DownloadResult:
    """download all requested symbols"""
    download: DownloadResult = download_with_passes(
        config,
        fetch_one=fetch_symbol_prices,
        fallback_frame=empty_price_frame(),
    )

    dataset: pd.DataFrame = download.dataset.sort_values(
        SYMBOL_DATE_KEY,
        kind="stable",
        ignore_index=True,
    )
    dataset = validate_frame(dataset, raw_prices_schema(), "raw prices")

    return DownloadResult(dataset=dataset, failed_ids=download.failed_ids)


def save_outputs(
    prices: pd.DataFrame, failed_symbols: Sequence[str], config: AppConfig
) -> None:
    """save the output files"""
    if prices.empty:
        raise RuntimeError(
            f"no price data returned for symbols={config.download_item_ids}; "
            f"failed_symbols={failed_symbols}"
        )

    prices = add_series_id_columns(prices)
    prices = finalize_grammar_frame(prices)
    saved_count: int = save_keyed_parquet_files(
        prices, SERIES_SYMBOL, config.output_dir
    )
    update_status(f"saved {saved_count} symbol files to {config.output_dir}")

    if failed_symbols:
        save_csv(
            pd.DataFrame({SERIES_SYMBOL: failed_symbols}),
            config.failed_symbols_path,
        )
        update_status(
            f"saved {len(failed_symbols)} failed symbols to {config.failed_symbols_path}"
        )


def main() -> None:
    """run the alpha vantage loader"""
    config: AppConfig = build_app_config(parse_arguments())

    update_status(f"resolved {len(config.download_item_ids)} symbols")
    download: DownloadResult = download_prices(config)
    save_outputs(download.dataset, download.failed_ids, config)


if __name__ == "__main__":
    main()
