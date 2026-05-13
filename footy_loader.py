import io
import warnings
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urljoin

import pandas as pd
from pandas import json_normalize

from utilities import (
    BaseAppConfig,
    JSONRequest,
    add_api_key_args,
    add_series_id_columns,
    clean_names,
    fetch_json_retries,
    finalize_grammar_frame,
    make_save_to_parser,
    parallelize_df,
    redact_mapping,
    resolve_api_key,
    save_csv,
    save_parquet,
    smart_type,
    uppercase_series_columns,
    update_status,
)

FOOTYSTATS_API_KEY_ENV: Final[str] = "FOOTYSTATS_API_KEY"
FOOTY_URL: Final[str] = "https://api.football-data-api.com/"


@dataclass(frozen=True, slots=True)
class AppConfig(BaseAppConfig):
    """hold runtime settings"""

    api_key: str
    base_url: str = FOOTY_URL

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError(
                "missing footystats api key; use --api_key, --api_key_file, "
                f"or set {FOOTYSTATS_API_KEY_ENV}"
            )
        BaseAppConfig.__post_init__(self)


def get_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """wrapper for requests"""
    json_request: JSONRequest = JSONRequest(endpoint, params=params)
    json_data = fetch_json_retries(json_request)
    if json_data is None:
        redacted_params: dict[str, Any] = redact_mapping(params)
        update_status(
            f"{endpoint} at {json_request.url}: no json returned with "
            f"{redacted_params}"
        )
        return {}

    if "data" not in json_data:
        redacted_params = redact_mapping(params)
        update_status(
            f"{endpoint} at {json_request.url}: no 'data' element found with "
            f"{redacted_params}"
        )
        return {}

    return json_data


def fetch_match_details(items: tuple[str, int, str, str]) -> pd.DataFrame:
    """fetch match details for a single season"""
    league_name, season_id, api_key, base_url = items
    endpoint: str = urljoin(base_url, "league-matches")
    params: dict[str, Any] = {"key": api_key, "season_id": season_id}

    json_data: dict[str, Any] = get_json(endpoint, params)
    if not json_data:
        update_status(f"{league_name} {season_id}: no json")
        return pd.DataFrame()

    ff: pd.DataFrame = pd.DataFrame(json_data["data"])
    if ff.empty:
        update_status(f"{league_name} {season_id}: no data")
        return pd.DataFrame()

    base_columns: dict[str, Any] = {x: ff[x].dtype for x in ff.columns}
    max_pages: int = json_data["pager"]["max_page"]
    dataset: list[pd.DataFrame] = [ff]
    for page in range(2, max_pages + 1):
        params["page"] = page
        json_data = get_json(endpoint, params)
        if not json_data:
            update_status(f"{league_name} {season_id}: no data at page {page}")
            continue
        pf: pd.DataFrame = pd.DataFrame(json_data["data"])

        if pf.empty:
            update_status(
                f"{league_name} {season_id}: no data at page {page} after build"
            )
            continue
        dataset.append(pf)

    if len(dataset) == 0:
        update_status(f"{league_name} {season_id}: dataset is empty - shd not happen")
        return pd.DataFrame()

    dataset = [x for x in dataset if not x.empty]
    if len(dataset) == 0:
        update_status(
            f"{league_name} {season_id}: dataset is full of empty frames - shd not happen"
        )
        return pd.DataFrame()

    dataset = [x.dropna(how="all").astype(base_columns) for x in dataset]
    if len(dataset) == 0:
        update_status(f"{league_name} {season_id}: dataset is full of nan frames")
        return pd.DataFrame()

    with warnings.catch_warnings(record=True) as w:
        df: pd.DataFrame = pd.concat(dataset, ignore_index=True)
        if w:
            warning_msg = ",".join([str(x.message) for x in w])
            update_status(
                f"{league_name} {season_id}: shape {df.shape} warning {warning_msg}"
            )
            return pd.DataFrame()

    if df.empty:
        update_status(
            f"{league_name} {season_id}: concatenated df is empty - shd not happen"
        )
        return pd.DataFrame()
    # stops fragmentation warning
    df = df.copy()
    df.insert(0, "CATEGORY_TEAM_DIVISION", league_name)
    df.insert(0, "SEASON_ID", season_id)
    df.columns = df.columns.str.upper()
    return df


def get_season_ids(config: AppConfig) -> pd.DataFrame:
    """fetch available league season ids and save to file"""
    endpoint: str = urljoin(config.base_url, "league-list")
    params: dict = {"key": config.api_key, "chosen_leagues_only": "true"}
    json_data: dict = get_json(endpoint, params)
    if not json_data:
        update_status(f"""{endpoint}: json is empty {params}""")
        return pd.DataFrame()
    df: pd.DataFrame = json_normalize(
        json_data["data"], record_path="season", meta=["name"]
    )
    df.columns = df.columns.str.upper()
    return df.rename(columns={"ID": "SEASON_ID"})


def grammar_name(column: str, dtype: Any) -> str:
    """map one footy source column into the engine grammar"""
    if column.startswith("TARGET_"):
        return f"FEATURE_{column.removeprefix('TARGET_')}"
    if column.startswith(("LABEL_", "FEATURE_", "IS_", "CATEGORY_", "SERIES_")):
        return column
    if column in {"HOME_NAME", "HOME", "TEAM_A"}:
        return "SERIES_HOME"
    if column in {"AWAY_NAME", "AWAY", "TEAM_B"}:
        return "SERIES_AWAY"
    if column in {"HOMEID", "HOME_ID", "TEAM_A_ID"}:
        return "LABEL_SOURCE_HOME_ID"
    if column in {"AWAYID", "AWAY_ID", "TEAM_B_ID"}:
        return "LABEL_SOURCE_AWAY_ID"
    if "DATE" in column or "TIME" in column:
        return f"LABEL_{column}"
    if column in {"ID", "MATCH_ID", "GAME_ID"}:
        return f"LABEL_{column}"
    if "SEASON" in column or "LEAGUE" in column or "DIVISION" in column:
        return f"CATEGORY_{column}"
    if column.startswith(("IS_", "HAS_")) or str(dtype) == "bool":
        return f"IS_{column.removeprefix('IS_')}"
    if any(token in column for token in ("SCORE", "GOAL", "POINT", "RESULT")):
        return f"FEATURE_{column}"
    if pd.api.types.is_numeric_dtype(dtype):
        return f"FEATURE_{column}"
    return f"CATEGORY_{column}"


def normalize_grammar(df: pd.DataFrame) -> pd.DataFrame:
    """normalize footy columns to the engine grammar"""
    rename_map: dict[str, str] = {
        column: grammar_name(column, df[column].dtype) for column in df.columns
    }
    df = df.rename(columns=rename_map)
    df = uppercase_series_columns(df)
    df = add_series_id_columns(df)
    return finalize_grammar_frame(df)


def parse_arguments() -> Namespace:
    """arg parser"""
    parser: ArgumentParser = make_save_to_parser(
        "download & normalize footy data",
        save_to_help="output directory",
    )
    add_api_key_args(parser, "footystats")
    return parser.parse_args()


def build_app_config(args: Namespace) -> AppConfig:
    """build the app config"""
    return AppConfig(
        api_key=resolve_api_key(
            args.api_key,
            args.api_key_file,
            FOOTYSTATS_API_KEY_ENV,
            "footystats",
        ),
        save_to=args.save_to,
    )


def main() -> None:
    """download footy_stats football data"""
    config: AppConfig = build_app_config(parse_arguments())

    season_ids: pd.DataFrame = get_season_ids(config)
    if season_ids.empty:
        raise RuntimeError("season ids are empty")
    season_file = config.output_dir / "footy_stats_seasons.csv"
    update_status(
        f"saving: {season_file}, shape: {season_ids.shape}, "
        f"leagues: {season_ids['NAME'].nunique()}"
    )
    save_csv(season_ids, season_file)
    seasons_to_fetch: list = [
        (name, season_id, config.api_key, config.base_url)
        for name, season_id in zip(season_ids["NAME"], season_ids["SEASON_ID"])
    ]
    match_frame: pd.DataFrame = parallelize_df(fetch_match_details, seasons_to_fetch)
    if match_frame.empty:
        raise RuntimeError("match frame is empty")

    match_frame["HOME_NAME"] = clean_names(match_frame["HOME_NAME"])
    match_frame["AWAY_NAME"] = clean_names(match_frame["AWAY_NAME"])
    match_frame["CATEGORY_TEAM_DIVISION"] = clean_names(
        match_frame["CATEGORY_TEAM_DIVISION"]
    )
    match_frame = smart_type(match_frame)
    match_frame = normalize_grammar(match_frame)

    save_file = config.output_dir / "all_matches.pqt"
    update_status(f"saving: {save_file}, shape: {match_frame.shape}")
    info_buffer: io.StringIO = io.StringIO()
    match_frame.info(buf=info_buffer, verbose=True)
    save_parquet(match_frame, save_file)
    update_status(info_buffer.getvalue())


if __name__ == "__main__":
    main()
