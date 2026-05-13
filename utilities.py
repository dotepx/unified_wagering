from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import ceil
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, TypeVar

import pandas as pd
import pandera.pandas as pa
import psutil
import pyarrow.parquet as pq
import requests
from pandera.errors import SchemaError, SchemaErrors
from requests.adapters import HTTPAdapter
from urllib3 import Retry

MAX_CPU: int = psutil.cpu_count(logical=False) or 1
START_TIME: datetime = datetime.now()
KENPOM_API: str = os.environ.get("KENPOM_API", os.environ.get("KENPOM_API_KEY", ""))
T = TypeVar("T")


class DownloadPassConfig(Protocol):
    """describe config needed for paced download passes"""

    @property
    def download_passes(self) -> int:
        """return number of download passes"""

    @property
    def request_spacing(self) -> float:
        """return seconds between request starts"""

    @property
    def download_item_ids(self) -> Sequence[str]:
        """return ids to download"""

    @property
    def item_label(self) -> str:
        """return singular item label for logging"""


D = TypeVar("D", bound=DownloadPassConfig)

GRAMMAR_PREFIXES: tuple[str, ...] = (
    "LABEL_",
    "FEATURE_",
    "TARGET_",
    "IS_",
    "CATEGORY_",
    "SERIES_",
)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
)


class SeriesType(Enum):
    """enumerated series type"""

    FOOTY = "footy"
    NCAA = "ncaa"
    STOCKS = "stocks"
    TENNIS = "tennis"


class ElapsedUnits(Enum):
    """enumerated elapsed time granularity"""

    USECS = 1e6
    MILLISECONDS = 1e3
    SECONDS = 1.0
    MINUTES = 1.0 / 60.0


class LogLevel(Enum):
    """enumerated log levels"""

    INFO = 1
    FATAL = 2
    ERROR = 3


@dataclass(frozen=True, slots=True)
class BaseAppConfig:
    """hold shared output settings for loader configs"""

    save_to: Path

    def __post_init__(self) -> None:
        self.save_to.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        """return the output directory"""
        return self.save_to


def make_save_to_parser(
    description: str,
    save_to_help: str = "output path",
) -> ArgumentParser:
    """create a standard argument parser with save_to"""
    parser: ArgumentParser = ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_save_to_arg(parser, save_to_help)
    return parser


def add_save_to_arg(
    parser: ArgumentParser,
    help_text: str = "output path",
) -> ArgumentParser:
    """add the standard save_to argument"""
    parser.add_argument(
        "-s",
        "--save_to",
        type=Path,
        required=True,
        help=help_text,
    )
    return parser


def add_api_key_args(
    parser: ArgumentParser,
    provider_name: str,
) -> ArgumentParser:
    """add standard api key arguments"""
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help=f"{provider_name} api key",
    )
    parser.add_argument(
        "--api_key_file",
        type=Path,
        default=None,
        help=f"text file that contains only the {provider_name} api key",
    )
    return parser


@dataclass(frozen=True)
class JSONRequest:
    """wrap configuration for one json api request"""

    url: str
    params: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    expected_key: Optional[str] = None
    timeout: int | tuple[float, float] = 10


@dataclass(frozen=True, slots=True)
class FetchedFrame:
    """hold one fetched frame"""

    item_id: str
    frame: pd.DataFrame
    cooldown_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """hold a downloaded frame collection"""

    dataset: pd.DataFrame
    failed_ids: tuple[str, ...]


def get_max_cpu() -> int:
    """return max permissible cpus"""
    return MAX_CPU


def get_gpu_memory() -> int:
    """query nvidia-smi for total vram in mb"""
    cmd: str = "nvidia-smi --query-gpu=memory.total --format=csv"
    try:
        out: str = subprocess.check_output(cmd.split()).decode("ascii")
        val: str = out.splitlines()[1].split()[0]
        return int(val)
    except Exception:
        return 0


def smart_type(df: pd.DataFrame) -> pd.DataFrame:
    """downcast to memory efficient types"""
    for col in [
        x
        for x in df.columns
        if pd.api.types.is_numeric_dtype(df[x].dtype)
        or pd.api.types.is_bool_dtype(df[x].dtype)
    ]:
        col_data: pd.Series = df[col]
        if pd.api.types.is_bool_dtype(col_data.dtype):
            df[col] = col_data.astype("uint8")
        elif pd.api.types.is_integer_dtype(col_data) or (col_data % 1 == 0).all():
            df[col] = pd.to_numeric(
                col_data, downcast="unsigned" if col_data.min() >= 0 else "integer"
            )
        elif df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        else:
            df[col] = pd.to_numeric(col_data, downcast="float")
    return df


def parallelize_df(
    func: Callable[[Any], pd.DataFrame],
    array: Sequence[Any],
    axis: Literal[0, 1] = 0,
    max_cpu: int = MAX_CPU,
) -> pd.DataFrame:
    """parallelize dataframe construction"""
    dataset: list[pd.DataFrame] = [
        x for x in parallelize(func, array, max_cpu) if not x.empty
    ]
    return (
        pd.concat(dataset, axis=axis, ignore_index=(axis == 0))
        if dataset
        else pd.DataFrame()
    )


def parallelize(
    func: Callable[[Any], T],
    array: Sequence[Any],
    max_cpu: int = MAX_CPU,
) -> list[T]:
    """parallelize a function over an array"""
    if not array:
        return []
    num_procs: int = min(max_cpu, len(array))
    if num_procs <= 1:
        return [func(array[0])]
    with Pool(processes=num_procs) as pool:
        return list(pool.imap_unordered(func, array))


def get_seed() -> int:
    """return shared random seed"""
    return 42


def get_memory() -> str:
    """return memory consumption"""
    units: float = 1024**2
    memory: float = ceil(psutil.virtual_memory().used / units)
    swap: float = ceil(psutil.swap_memory().used / units)
    return f"memory: {memory}MB, swap: {swap}MB"


def get_cpu_utilization() -> str:
    """return cpu utilization"""
    cpu_usage: float = psutil.cpu_percent(interval=None)
    return f"cpu: {cpu_usage}"


def get_elapsed_time(units: ElapsedUnits = ElapsedUnits.SECONDS) -> str:
    """return elapsed time between start time and now"""
    elapsed: int = int(
        ceil((datetime.now() - START_TIME).total_seconds() * units.value)
    )
    return f"{units.name.lower()}: {elapsed}"


def get_elapased_time(units: ElapsedUnits = ElapsedUnits.SECONDS) -> str:
    """return elapsed time with legacy misspelled name"""
    return get_elapsed_time(units)


def update_status(
    msg: str,
    level: LogLevel = LogLevel.INFO,
    units: ElapsedUnits = ElapsedUnits.SECONDS,
) -> None:
    """log one status message"""
    info: str = (
        f"[{get_elapsed_time(units)}, {get_memory()}, {get_cpu_utilization()}]->"
    )
    if level == LogLevel.FATAL:
        logging.fatal(f"{info} FATAL: {msg}")
        sys.exit(1)
    if level == LogLevel.ERROR:
        logging.error(f"{info} ERROR: {msg}")
        return
    logging.info(f"{info} {msg}")


def clean_names(ds: pd.Series) -> pd.Series:
    """clean entity names"""
    ds = ds.str.upper()
    ds = ds.replace(r"\*+", "", regex=True)
    ds = ds.replace(r"\.+", " ", regex=True)
    ds = ds.replace(r"\'+", "", regex=True)
    ds = ds.replace(r"\d+", " ", regex=True)
    ds = ds.replace(r"\-+", " ", regex=True)
    ds = ds.replace(r"\s+", " ", regex=True)
    return ds.str.strip()


def split_cli_tokens(raw_values: Sequence[str]) -> list[str]:
    """split comma-delimited cli tokens"""
    return [
        token.strip()
        for raw_value in raw_values
        for token in raw_value.split(",")
        if token.strip()
    ]


def read_first_column_csv(path: Path, uppercase: bool = False) -> list[str]:
    """read the first csv column as ids"""
    values: pd.Series = pd.read_csv(path).iloc[:, 0].dropna().astype(str).str.strip()
    values = values.str.upper() if uppercase else values
    return values.loc[lambda series: series.ne("")].drop_duplicates().tolist()


def resolve_identifiers(
    raw_values: Sequence[str],
    csv_path: Path | None,
    uppercase: bool = False,
) -> tuple[str, ...]:
    """resolve ids from cli and csv"""
    cli_ids: list[str] = (
        [
            token.upper() if uppercase else token
            for token in split_cli_tokens(raw_values)
        ]
        if raw_values
        else []
    )
    csv_ids: list[str] = (
        read_first_column_csv(csv_path, uppercase=uppercase) if csv_path else []
    )
    return tuple(sorted(set(cli_ids + csv_ids)))


def resolve_api_key(
    cli_key: str,
    api_key_file: Path | None,
    env_name: str,
    provider_name: str,
) -> str:
    """resolve an api key from cli, file, or env"""
    if cli_key.strip():
        return cli_key.strip()
    if api_key_file:
        file_key: str = api_key_file.read_text(encoding="utf-8").strip()
        if file_key:
            return file_key
        raise ValueError(f"api key file is empty: {api_key_file}")
    env_key: str = os.environ.get(env_name, "").strip()
    if env_key:
        return env_key
    raise ValueError(
        f"missing {provider_name} api key; use --api_key, --api_key_file, or set {env_name}"
    )


def redact_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    """redact sensitive values from loggable mappings"""
    secret_tokens: tuple[str, ...] = ("key", "token", "secret", "authorization")
    return {
        key: (
            "***REDACTED***"
            if any(token in key.lower() for token in secret_tokens)
            else value
        )
        for key, value in values.items()
    }


def validate_frame(
    df: pd.DataFrame,
    schema: Any,
    frame_name: str,
) -> pd.DataFrame:
    """validate one dataframe"""
    if schema is None:
        return df
    try:
        return schema.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as ex:
        failure_cases: Any = getattr(ex, "failure_cases", None)
        details: str = (
            f"\n{failure_cases.head(10).to_string(index=False)}"
            if failure_cases is not None
            else ""
        )
        raise ValueError(f"{frame_name} failed pandera validation{details}") from ex


def download_with_passes(
    config: D,
    fetch_one: Callable[[str, D], FetchedFrame],
    fallback_frame: pd.DataFrame | None = None,
) -> DownloadResult:
    """download items across paced retry passes"""
    item_ids: Sequence[str] = config.download_item_ids
    dataset: list[pd.DataFrame] = []
    remaining_ids: Sequence[str] = item_ids
    next_request_at: float = time.monotonic()

    for pass_num in range(1, config.download_passes + 1):
        if not remaining_ids:
            break
        update_status(
            f"starting {config.item_label} pass "
            f"{pass_num}/{config.download_passes} for {len(remaining_ids)} ids"
        )
        failed_ids: list[str] = []
        for idx, item_id in enumerate(remaining_ids, start=1):
            wait_time: float = next_request_at - time.monotonic()
            if wait_time > 0.0:
                time.sleep(wait_time)
            update_status(
                f"fetching {config.item_label} pass {pass_num}/{config.download_passes} "
                f"{idx}/{len(remaining_ids)}: {item_id}"
            )
            result: FetchedFrame = fetch_one(item_id, config)
            if not result.frame.empty:
                dataset.append(result.frame)
            if result.frame.empty:
                failed_ids.append(result.item_id)
            if result.cooldown_seconds > 0.0:
                update_status(
                    f"cooling down {result.cooldown_seconds:.1f}s after "
                    f"{config.item_label} {result.item_id}"
                )
                time.sleep(result.cooldown_seconds)
            next_request_at = max(
                next_request_at + config.request_spacing,
                time.monotonic(),
            )
        remaining_ids = failed_ids

    if dataset:
        result_frame: pd.DataFrame = pd.concat(dataset, axis=0, ignore_index=True)
    elif fallback_frame is not None:
        result_frame = fallback_frame.copy()
    else:
        result_frame = pd.DataFrame()

    return DownloadResult(
        dataset=result_frame,
        failed_ids=tuple(sorted(set(remaining_ids))),
    )


def get_retry_session(
    retries: int = 3,
    backoff_factor: int = 1,
    status_forcelist: Sequence[int] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """create a requests session with automatic retries"""
    session: requests.Session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


HTTP: requests.Session = get_retry_session()


def fetch_json(config: JSONRequest) -> Any:
    """fetch json data from a url"""
    try:
        response = requests.get(
            config.url,
            params=config.params,
            headers=config.headers,
            timeout=config.timeout,
        )
        response.raise_for_status()
        return response.json()
    except Exception as ex:
        update_status(f"unable to get json at {config.url} because {ex}")
        return None


def fetch_json_retries(config: JSONRequest) -> Any:
    """fetch json data from a url with retries"""
    try:
        response = HTTP.get(
            config.url,
            params=config.params,
            headers=config.headers,
            timeout=config.timeout,
        )
        response.raise_for_status()
        return response.json()
    except Exception as ex:
        update_status(
            f"error fetching {config.url} params={redact_mapping(config.params)}: {ex}"
        )
        return None


def _normalize_id_columns(columns: Sequence[str] | str) -> tuple[str, ...]:
    """normalize one or more id source columns"""
    return (columns,) if isinstance(columns, str) else tuple(columns)


def _is_id_source_dtype(dtype: Any) -> bool:
    """return whether a dtype can be used as a stable id source"""
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def _validate_id_source_columns(
    df: pd.DataFrame,
    columns: Sequence[str] | str,
) -> tuple[str, ...]:
    """validate columns for stable id generation"""
    source_columns: tuple[str, ...] = _normalize_id_columns(columns)
    if not source_columns:
        raise ValueError("at least one id source column is required")
    missing_columns: list[str] = [
        column for column in source_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"missing id source columns: {missing_columns}")
    invalid_columns: list[str] = [
        column for column in source_columns if not _is_id_source_dtype(df[column].dtype)
    ]
    if invalid_columns:
        raise ValueError(
            f"id source columns must be string or object typed: {invalid_columns}"
        )
    return source_columns


def _clean_id_values(values: pd.Series) -> pd.Series:
    """normalize id source values"""
    cleaned_values: pd.Series = values.astype("string").str.strip()
    return cleaned_values.mask(cleaned_values.eq(""))


def make_stable_uint_id(value: str) -> int:
    """return a stable uint64 id for one normalized string value"""
    digest: bytes = hashlib.blake2b(
        value.encode("utf-8"),
        digest_size=8,
        person=b"seriesid",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def make_stable_uint_ids(values: pd.Series) -> pd.Series:
    """return stable uint64 ids for normalized string values"""
    return values.map(make_stable_uint_id).astype("uint64")


def make_id_map(
    df: pd.DataFrame,
    columns: Sequence[str] | str,
    value_column: str = "SERIES_VALUE",
    id_column: str = "SERIES_VALUE_ID",
) -> pd.DataFrame:
    """build a stable string or object id map"""
    source_columns: tuple[str, ...] = _validate_id_source_columns(df, columns)
    values: pd.Series = pd.concat(
        [df[column] for column in source_columns],
        ignore_index=True,
    )
    values = _clean_id_values(values).dropna()
    if values.empty:
        raise ValueError("no non-empty values available for id map")

    id_map: pd.DataFrame = pd.DataFrame(
        {value_column: sorted(values.drop_duplicates().tolist())}
    )
    id_map[id_column] = make_stable_uint_ids(id_map[value_column])
    duplicate_ids: pd.DataFrame = id_map.loc[id_map.duplicated(id_column, keep=False)]
    if not duplicate_ids.empty:
        raise ValueError(
            f"stable id collision in {id_column}: "
            f"{duplicate_ids.sort_values(id_column).to_dict(orient='records')}"
        )
    return id_map


def make_id_column_name(column: str) -> str:
    """build a grammar-compatible series id column name"""
    base_name: str = column.upper()
    for prefix in GRAMMAR_PREFIXES:
        if base_name.startswith(prefix):
            base_name = base_name.removeprefix(prefix)
            break
    base_name = base_name.removesuffix("_ID")
    if not base_name:
        raise ValueError(f"unable to derive id column name from {column!r}")
    return f"SERIES_{base_name}_ID"


def add_id_columns(df: pd.DataFrame, columns: Sequence[str] | str) -> pd.DataFrame:
    """append stable uint64 id columns for string or object columns"""
    source_columns: tuple[str, ...] = _validate_id_source_columns(df, columns)
    result: pd.DataFrame = df.copy()
    id_map: pd.DataFrame = make_id_map(result, source_columns)
    lookup: dict[str, int] = dict(
        zip(id_map["SERIES_VALUE"], id_map["SERIES_VALUE_ID"])
    )
    id_dtype: Any = id_map["SERIES_VALUE_ID"].dtype
    for column in source_columns:
        id_column: str = make_id_column_name(column)
        if id_column in result.columns:
            raise ValueError(f"id column already exists: {id_column}")
        cleaned_values: pd.Series = _clean_id_values(result[column])
        if cleaned_values.isna().any():
            raise ValueError(f"id source column contains empty values: {column}")
        ids: pd.Series = cleaned_values.map(lookup)
        if ids.isna().any():
            raise ValueError(f"unable to map all values for {column}")
        result[id_column] = ids.astype(id_dtype)
    return result


def add_series_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """append stable uint64 ids for all string series columns"""
    source_columns: tuple[str, ...] = tuple(
        column
        for column in df.columns
        if column.startswith("SERIES_") and _is_id_source_dtype(df[column].dtype)
    )
    return add_id_columns(df, source_columns) if source_columns else df.copy()


def uppercase_series_columns(df: pd.DataFrame) -> pd.DataFrame:
    """uppercase all string series columns"""
    output: pd.DataFrame = df.copy()
    series_columns: list[str] = [
        column
        for column in output.columns
        if column.startswith("SERIES_") and _is_id_source_dtype(output[column].dtype)
    ]
    for column in series_columns:
        output[column] = output[column].astype("string").str.strip().str.upper()
    return output


def make_ids(df: pd.DataFrame, columns: Sequence[str], lbl: str) -> pd.DataFrame:
    """get distinct ids for column values with the legacy id name"""
    return make_id_map(
        df,
        columns,
        value_column=lbl,
        id_column=f"{lbl}ID",
    )


def make_file_stem(item_id: str) -> str:
    """build a filesystem-safe file stem"""
    stem: str = re.sub(r"[^A-Za-z0-9._-]+", "_", item_id.strip())
    return stem.strip("._") or "item"


def save_parquet(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """save one parquet file"""
    parquet_path: Path = Path(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        parquet_path,
        compression="zstd",
        index=index,
    )


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """save one csv file"""
    csv_path: Path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=index)


def save_parquet_task(task: tuple[pd.DataFrame, Path]) -> None:
    """save one dataframe and path task"""
    df, path = task
    save_parquet(df, path)


def save_keyed_parquet_files(df: pd.DataFrame, key: str, output_dir: Path) -> int:
    """save one parquet file per key value"""
    if key not in df.columns:
        raise ValueError(f"missing partition key {key!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[pd.DataFrame, Path]] = [
        (item_df, output_dir / f"{make_file_stem(str(item_id))}.pqt")
        for item_id, item_df in df.groupby(key, sort=False)
    ]
    task_count: int = len(tasks)

    if task_count == 1:
        save_parquet_task(tasks[0])
    elif task_count > 1:
        worker_count: int = min(MAX_CPU, task_count)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(save_parquet_task, tasks))

    return task_count


def load_parquet(path: str | Path) -> pd.DataFrame:
    """read a parquet file into a dataframe"""
    parquet_path: Path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"missing parquet file: {parquet_path}")
    return pq.read_table(parquet_path).to_pandas()


def load_json(input_path: str) -> dict[str, Any]:
    """load json as dict"""
    with open(input_path, "r", encoding="utf-8") as f:
        return dict(json.load(f))


def write_json(save_to: str, save_data: Mapping[str, Any]) -> None:
    """write dict as json"""
    with open(save_to, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)


def assert_grammar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """fail when dataframe columns are outside the engine grammar"""
    invalid_columns: list[str] = [
        column
        for column in df.columns
        if not any(column.startswith(prefix) for prefix in GRAMMAR_PREFIXES)
    ]
    if invalid_columns:
        raise ValueError(f"columns outside grammar: {invalid_columns}")
    return df


def ordered_grammar_columns(df: pd.DataFrame) -> list[str]:
    """return columns in stable grammar order"""
    return [
        column
        for prefix in GRAMMAR_PREFIXES
        for column in df.columns
        if column.startswith(prefix)
    ]


def finalize_grammar_frame(df: pd.DataFrame) -> pd.DataFrame:
    """order and validate a grammar-normalized dataframe"""
    return assert_grammar_columns(df.loc[:, ordered_grammar_columns(df)])
