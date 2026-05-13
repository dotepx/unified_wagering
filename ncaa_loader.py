#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import pandera.pandas as pa
import requests
from pandera.typing import Series

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
    resolve_api_key,
    save_parquet,
    smart_type,
    uppercase_series_columns,
    update_status,
)

KENPOM_API_KEY_ENV: str = "KENPOM_API"


class RefDataSchema(pa.DataFrameModel):
    """Schema for reference data after load_ref_data."""

    CATEGORY_SEASON: Series[int] = pa.Field(ge=2005, le=date.today().year + 1)
    TEAMNAME: Series[str] = pa.Field(str_length={"min_value": 1})
    TEAMID: Series[int] = pa.Field(ge=1)
    CONFERENCE: Series[str] = pa.Field(str_length={"min_value": 1})
    CONFERENCEID: Series[int] = pa.Field(ge=1)
    COACH: Series[str] = pa.Field(str_length={"min_value": 1})
    COACHID: Series[int] = pa.Field(ge=1)
    FINALNAME: Series[str] = pa.Field(str_length={"min_value": 1})
    CANONICALNAME: Series[str] = pa.Field(str_length={"min_value": 1})

    class Config:
        coerce = True
        strict = False


class AllNamesSchema(pa.DataFrameModel):
    """Schema for canonical name mapping from build_canonical_names."""

    TEAMNAME: Series[str] = pa.Field(str_length={"min_value": 1}, unique=True)
    CANONICALNAME: Series[str] = pa.Field(str_length={"min_value": 1})
    TEAMID: Series[np.uint32] = pa.Field(ge=1)

    class Config:
        coerce = True
        strict = True

    @pa.dataframe_check(name="canonical_id_consistency")
    @classmethod
    def canonical_maps_to_one_id(cls, df: pd.DataFrame) -> bool:
        """Each canonical name must map to exactly one TEAMID."""
        max_ids_per_name: int = df.groupby("CANONICALNAME")["TEAMID"].nunique().max()
        return max_ids_per_name == 1


class MatchDetailsSchema(pa.DataFrameModel):
    """Schema for raw match data (historical + scheduled)."""

    LABEL_DATE: Series[pd.Timestamp] = pa.Field(nullable=False)
    CATEGORY_SEASON: Series[int] = pa.Field(ge=2005, le=date.today().year + 1)
    HOME: Series[str] = pa.Field(str_length={"min_value": 1})
    AWAY: Series[str] = pa.Field(str_length={"min_value": 1})
    HOMESCORE: Series[np.int16] = pa.Field(ge=0)
    AWAYSCORE: Series[np.int16] = pa.Field(ge=0)
    IS_COMPLETE: Series[int] = pa.Field(isin=[0, 1])
    IS_NEUTRAL: Series[np.uint8] = pa.Field(isin=[0, 1])
    FEATURE_OVERTIME_PERIODS: Series[float] = pa.Field(ge=0.0)

    class Config:
        coerce = True
        strict = False


class MergedMatchSchema(pa.DataFrameModel):
    """Schema for matches after merge_refdata."""

    LABEL_DATE: Series[pd.Timestamp] = pa.Field(nullable=False)
    CATEGORY_SEASON: Series[int] = pa.Field(ge=2005)
    HOME: Series[str] = pa.Field(str_length={"min_value": 1})
    AWAY: Series[str] = pa.Field(str_length={"min_value": 1})
    HOMEID: Series[np.uint32] = pa.Field(ge=1)
    AWAYID: Series[np.uint32] = pa.Field(ge=1)
    HOME_CONFERENCE: Series[str] = pa.Field(str_length={"min_value": 1})
    AWAY_CONFERENCE: Series[str] = pa.Field(str_length={"min_value": 1})
    HOME_CONFERENCEID: Series[np.uint32] = pa.Field(ge=1)
    AWAY_CONFERENCEID: Series[np.uint32] = pa.Field(ge=1)
    HOME_COACH: Series[str] = pa.Field(str_length={"min_value": 1})
    AWAY_COACH: Series[str] = pa.Field(str_length={"min_value": 1})
    HOME_COACHID: Series[np.uint32] = pa.Field(ge=1)
    AWAY_COACHID: Series[np.uint32] = pa.Field(ge=1)
    IS_COMPLETE: Series[int] = pa.Field(isin=[0, 1])
    IS_NEUTRAL: Series[np.uint8] = pa.Field(isin=[0, 1])

    class Config:
        coerce = True
        strict = False

    @staticmethod
    def _check_id_pair(
        df: pd.DataFrame,
        label_a: str,
        id_a: str,
        label_b: str,
        id_b: str,
    ) -> bool:
        """Verify 1:1 mapping between labels and ids across home/away."""
        universe: pd.DataFrame = pd.concat(
            [
                df[[label_a, id_a]].rename(columns={label_a: "L", id_a: "I"}),
                df[[label_b, id_b]].rename(columns={label_b: "L", id_b: "I"}),
            ]
        ).drop_duplicates()
        max_ids_per_label: int = universe.groupby("L")["I"].nunique().max()
        max_labels_per_id: int = universe.groupby("I")["L"].nunique().max()
        return max_ids_per_label == 1 and max_labels_per_id == 1

    @pa.dataframe_check(name="no_self_play")
    @classmethod
    def home_away_differ(cls, df: pd.DataFrame) -> bool:
        """A team cannot play itself."""
        return bool((df["HOMEID"] != df["AWAYID"]).all())

    @pa.dataframe_check(name="team_id_bijection")
    @classmethod
    def team_ids_consistent(cls, df: pd.DataFrame) -> bool:
        """Each team name maps to exactly one id and vice versa."""
        return cls._check_id_pair(df, "HOME", "HOMEID", "AWAY", "AWAYID")

    @pa.dataframe_check(name="conference_id_bijection")
    @classmethod
    def conf_ids_consistent(cls, df: pd.DataFrame) -> bool:
        """Each conference name maps to exactly one id and vice versa."""
        return cls._check_id_pair(
            df,
            "HOME_CONFERENCE",
            "HOME_CONFERENCEID",
            "AWAY_CONFERENCE",
            "AWAY_CONFERENCEID",
        )


@dataclass(frozen=True, slots=True)
class KenPom:
    """thin wrapper around the kenpom api configuration"""

    api_key: str
    base_url: str = "https://kenpom.com"
    api: str = "api.php"
    start_year: int = 2005
    end_year: int = date.today().year + 1
    seasons: tuple[int, ...] = tuple(range(start_year, end_year))

    @property
    def headers(self) -> dict[str, str]:
        """return authorization headers"""
        return {"Authorization": f"Bearer {self.api_key}"}


KENPOM: KenPom = KenPom(api_key="")


@dataclass(frozen=True, slots=True)
class AppConfig(BaseAppConfig):
    """hold runtime settings"""

    games: str
    clean_names: str
    match_date: str
    api_key: str
    api_key_file: Path | None = None

    def __post_init__(self) -> None:
        pd.to_datetime(self.match_date, format="%Y-%m-%d", errors="raise")
        BaseAppConfig.__post_init__(self)


def get_ref_data_by_season(season: int) -> pd.DataFrame:
    """Fetch KenPom reference data for one season."""
    config: JSONRequest = JSONRequest(
        url=urljoin(KENPOM.base_url, KENPOM.api),
        params={"endpoint": "teams", "y": season},
        headers=KENPOM.headers,
        timeout=10,
    )
    try:
        result: pd.DataFrame = pd.DataFrame(fetch_json_retries(config))
        return result
    except Exception as ex:
        update_status(f"Error processing json for season {season}: {ex}")
        return pd.DataFrame()


def load_ref_data(clean_xref: pd.DataFrame) -> pd.DataFrame:
    """Load team reference data and derive canonical names."""
    refdata: pd.DataFrame = parallelize_df(get_ref_data_by_season, KENPOM.seasons)
    final_teamname: pd.DataFrame

    refdata.columns = refdata.columns.str.upper()
    refdata["COACH"] = clean_names(refdata["COACH"])
    refdata["COACHID"] = refdata["COACH"].factorize()[0] + 1
    refdata["CONFERENCE"] = refdata["CONFSHORT"].str.upper()
    refdata["CONFERENCEID"] = refdata["CONFSHORT"].factorize()[0] + 1
    refdata = refdata.rename(columns={"SEASON": "CATEGORY_SEASON"})[
        [
            "CATEGORY_SEASON",
            "TEAMNAME",
            "TEAMID",
            "CONFERENCE",
            "CONFERENCEID",
            "COACH",
            "COACHID",
        ]
    ]

    final_teamname = (
        refdata.sort_values("CATEGORY_SEASON")
        .groupby("TEAMID", as_index=False)
        .agg({"TEAMNAME": "last"})
        .rename(columns={"TEAMNAME": "FINALNAME"})
    )
    refdata = refdata.merge(final_teamname, on=["TEAMID"], how="left", validate="m:1")
    refdata["FINALNAME"] = clean_names(refdata["FINALNAME"])
    refdata = refdata.merge(
        clean_xref.rename(columns={"TEAMNAME": "FINALNAME"}),
        on="FINALNAME",
        how="left",
        validate="m:1",
    ).fillna({"CANONICALNAME": refdata["FINALNAME"]})

    RefDataSchema.validate(refdata)
    return refdata


def get_match_details(season_id: int) -> pd.DataFrame:
    """Fetch historical match details for one NCAA season."""
    file_id: int = season_id - 2000
    season_label: str = f"{file_id:02d}"
    data_file: str = f"cbbga{season_label}.txt"
    url: str = urljoin(KENPOM.base_url, data_file)
    response: requests.Response
    content: str
    type_map: dict[str, type] = {
        "DATE": str,
        "AWAY": str,
        "AWAYSCORE": np.int16,
        "HOME": str,
        "HOMESCORE": np.int16,
        "STATUS": str,
    }
    neutral_pattern: re.Pattern[str] = re.compile(r"^\d*N")
    matches: pd.DataFrame

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except Exception as ex:
        update_status(f"error fetching {url}: {ex}")
        return pd.DataFrame()

    content = response.content.decode("utf-8", errors="replace")
    try:
        matches = pd.read_fwf(
            io.StringIO(content),
            widths=[10, 24, 3, 24, 3, 24],
            names=list(type_map.keys()),
            dtype=type_map,
        )
        matches = matches.rename(columns={"DATE": "LABEL_DATE"})
        matches["STATUS"] = matches["STATUS"].fillna("").str.upper()
        matches["IS_COMPLETE"] = 1
        matches["IS_NEUTRAL"] = (
            matches["STATUS"].str.match(neutral_pattern, na=False).astype("uint8")
        )
        matches["FEATURE_OVERTIME_PERIODS"] = pd.to_numeric(
            matches["STATUS"].str.extract(r"^(\d+)", expand=False),
            errors="coerce",
        ).fillna(0)
        matches["LABEL_DATE"] = pd.to_datetime(matches["LABEL_DATE"], errors="coerce")
        matches["CATEGORY_SEASON"] = season_id
        return matches.drop(columns="STATUS")
    except Exception as ex:
        update_status(f"error parsing data for {data_file}: {ex}")
        return pd.DataFrame()


def load_historical_matches() -> pd.DataFrame:
    """Fetch historical match data across all seasons."""
    season_data: pd.DataFrame = parallelize_df(get_match_details, KENPOM.seasons)
    if season_data.empty:
        update_status("no historical match data")
    return season_data


def load_todays_matches(config: AppConfig, conferences: list[str]) -> pd.DataFrame:
    """Load today's games from csv and normalize match text."""
    matches: pd.DataFrame = pd.read_csv(config.games)
    sorted_conferences: list[str] = sorted(conferences, key=len, reverse=True)
    escaped_conferences: list[str] = [re.escape(conf) for conf in sorted_conferences]
    conference_pattern: str = rf"\s\b(?:{'|'.join(escaped_conferences)})(?:-T)?$"
    match_date: pd.Timestamp = pd.to_datetime(config.match_date, errors="coerce")

    matches["Game"] = matches["Game"].str.replace(
        conference_pattern,
        "",
        regex=True,
        flags=re.IGNORECASE,
    )
    matches["Game"] = matches["Game"].str.replace(r"[0-9]+", "", regex=True)
    matches["Game"] = matches["Game"].str.replace(r"\s+", " ", regex=True)
    matches["Game"] = matches["Game"].str.replace("NR ", "")
    matches["Game"] = matches["Game"].str.replace(" NCAA", "")
    matches["Game"] = matches["Game"].str.replace(r"\b\w*-T\b", "", regex=True)
    matches["Game"] = matches["Game"].str.strip()
    matches["IS_COMPLETE"] = 0
    matches["IS_NEUTRAL"] = (matches["Game"].str.find(" vs. ") >= 0).astype("uint8")
    matches["FEATURE_OVERTIME_PERIODS"] = 0.0
    matches["Game"] = matches["Game"].str.replace(" at ", "@")
    matches["Game"] = matches["Game"].str.replace(" vs. ", "@", regex=False)
    matches[["AWAY", "HOME"]] = matches["Game"].str.split("@", expand=True)
    matches["AWAY"] = matches["AWAY"].str.strip()
    matches["HOME"] = matches["HOME"].str.strip()
    matches["LABEL_DATE"] = match_date
    matches["CATEGORY_SEASON"] = match_date.year + int(match_date.month >= 8)
    matches[["AWAYSCORE", "HOMESCORE"]] = [0, 0]
    return matches.drop(columns=["Game"])


def build_match_details(refdata: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Build combined historical and scheduled match details."""
    historical_matches: pd.DataFrame = load_historical_matches()
    scheduled_matches: pd.DataFrame = load_todays_matches(
        config,
        list(refdata["CONFERENCE"].unique()),
    )
    all_matches: pd.DataFrame = pd.concat(
        [historical_matches, scheduled_matches[historical_matches.columns]],
        ignore_index=True,
    )
    all_matches["LABEL_DATE"] = all_matches["LABEL_DATE"].dt.normalize()
    MatchDetailsSchema.validate(all_matches)
    return all_matches


def build_canonical_names(
    all_matches: pd.DataFrame,
    clean_xref: pd.DataFrame,
    refdata: pd.DataFrame,
) -> pd.DataFrame:
    """Build canonical team names using refdata as the naming authority."""
    all_names: pd.DataFrame = pd.DataFrame(
        {"TEAMNAME": pd.concat([all_matches["HOME"], all_matches["AWAY"]]).unique()}
    )
    clean_map: pd.Series = clean_xref.set_index("TEAMNAME")["CANONICALNAME"]
    ref_name_map: pd.Series = (
        refdata[["TEAMNAME", "CANONICALNAME"]]
        .drop_duplicates()
        .assign(CLEAN=lambda frame: clean_names(frame["TEAMNAME"]))
        .set_index("CLEAN")["CANONICALNAME"]
    )
    teamid_map: pd.Series = (
        refdata[["CANONICALNAME", "TEAMID"]]
        .drop_duplicates(subset="CANONICALNAME")
        .set_index("CANONICALNAME")["TEAMID"]
    )
    missing_names: np.ndarray
    max_team_id: int
    missing_id_map: dict[str, int]
    missing_mask: pd.Series
    result: pd.DataFrame

    all_names["CLEANNAME"] = clean_names(all_names["TEAMNAME"])
    all_names["CANONICALNAME"] = (
        all_names["CLEANNAME"]
        .map(clean_map)
        .fillna(all_names["CLEANNAME"].map(ref_name_map))
        .fillna(all_names["CLEANNAME"])
    )
    all_names["TEAMID"] = all_names["CANONICALNAME"].map(teamid_map)

    missing_names = all_names.loc[all_names["TEAMID"].isna(), "CANONICALNAME"].unique()
    if missing_names.size > 0:
        max_team_id = int(all_names["TEAMID"].max())
        missing_id_map = {
            canonical_name: max_team_id + offset + 1
            for offset, canonical_name in enumerate(sorted(missing_names))
        }
        missing_mask = all_names["TEAMID"].isna()
        all_names.loc[missing_mask, "TEAMID"] = all_names.loc[
            missing_mask,
            "CANONICALNAME",
        ].map(missing_id_map)

    all_names["TEAMID"] = all_names["TEAMID"].astype("uint32")
    result = all_names.drop(columns=["CLEANNAME"])
    AllNamesSchema.validate(result)
    return result


def merge_refdata(
    all_matches: pd.DataFrame,
    all_names: pd.DataFrame,
    refdata: pd.DataFrame,
) -> pd.DataFrame:
    """Merge reference data into the match details dataset."""
    other_coach_id: int = int(refdata["COACHID"].max()) + 1
    other_conf_id: int = int(refdata["CONFERENCEID"].max()) + 1
    team_ref: pd.DataFrame = refdata.drop(columns=["TEAMNAME", "FINALNAME", "TEAMID"])
    side_label: str
    rename_map: dict[str, str]

    for side_label in ("HOME", "AWAY"):
        all_matches = (
            all_matches.merge(
                all_names.rename(columns={"TEAMNAME": side_label}),
                on=side_label,
                how="left",
                validate="m:1",
            )
            .drop(columns=[side_label])
            .rename(columns={"TEAMID": f"{side_label}ID", "CANONICALNAME": side_label})
        )

        rename_map = {
            "CANONICALNAME": side_label,
            "CONFERENCE": f"{side_label}_CONFERENCE",
            "CONFERENCEID": f"{side_label}_CONFERENCEID",
            "COACH": f"{side_label}_COACH",
            "COACHID": f"{side_label}_COACHID",
        }
        all_matches = all_matches.merge(
            team_ref.rename(columns=rename_map),
            on=["CATEGORY_SEASON", side_label],
            how="left",
            validate="m:1",
        )
        all_matches[f"{side_label}_CONFERENCE"] = all_matches[
            f"{side_label}_CONFERENCE"
        ].fillna("OTHER")
        all_matches[f"{side_label}_CONFERENCEID"] = all_matches[
            f"{side_label}_CONFERENCEID"
        ].fillna(other_conf_id)
        all_matches[f"{side_label}_COACH"] = all_matches[f"{side_label}_COACH"].fillna(
            "OTHER"
        )
        all_matches[f"{side_label}_COACHID"] = all_matches[
            f"{side_label}_COACHID"
        ].fillna(other_coach_id)

    MergedMatchSchema.validate(all_matches)
    return all_matches


def normalize_grammar(xf: pd.DataFrame) -> pd.DataFrame:
    """normalize ncaa match columns to the engine grammar"""
    df: pd.DataFrame = xf.rename(
        columns={
            "HOME": "SERIES_HOME",
            "AWAY": "SERIES_AWAY",
            "HOMEID": "LABEL_SOURCE_HOME_ID",
            "AWAYID": "LABEL_SOURCE_AWAY_ID",
            "HOMESCORE": "FEATURE_HOME_SCORE",
            "AWAYSCORE": "FEATURE_AWAY_SCORE",
            "HOME_CONFERENCE": "CATEGORY_HOME_CONFERENCE",
            "AWAY_CONFERENCE": "CATEGORY_AWAY_CONFERENCE",
            "HOME_CONFERENCEID": "CATEGORY_HOME_CONFERENCE_ID",
            "AWAY_CONFERENCEID": "CATEGORY_AWAY_CONFERENCE_ID",
            "HOME_COACH": "CATEGORY_HOME_COACH",
            "AWAY_COACH": "CATEGORY_AWAY_COACH",
            "HOME_COACHID": "CATEGORY_HOME_COACH_ID",
            "AWAY_COACHID": "CATEGORY_AWAY_COACH_ID",
        }
    )
    df = uppercase_series_columns(df)
    df = add_series_id_columns(df)
    return finalize_grammar_frame(df)


def finalize_and_save(xf: pd.DataFrame, config: AppConfig) -> None:
    """finalize the dataset and save it to parquet"""
    save_file: str = os.path.join(config.save_to, "all_matches.pqt")
    xf = normalize_grammar(
        smart_type(xf).astype(
            {"HOMESCORE": "int16", "AWAYSCORE": "int16"},
            errors="ignore",
        )
    )

    save_parquet(xf, save_file)
    update_status(str(xf.info(verbose=True)))
    update_status(
        f"last complete match date: {xf[xf['IS_COMPLETE'] > 0]['LABEL_DATE'].max()}"
    )


def parse_arguments() -> Namespace:
    """Parse command line arguments."""
    parser: ArgumentParser = make_save_to_parser(
        "generate model feature set",
        save_to_help="output directory",
    )

    parser.add_argument(
        "-g", "--games", type=str, required=True, help="today's matches path"
    )
    parser.add_argument(
        "-c", "--clean_names", type=str, required=True, help="team xref data path"
    )
    parser.add_argument(
        "-t",
        "--match_date",
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
        help="match date in YYYY-MM-DD format",
    )
    add_api_key_args(parser, "kenpom")
    return parser.parse_args()


def build_app_config(args: Namespace) -> AppConfig:
    """build the app config"""
    return AppConfig(
        games=args.games,
        save_to=args.save_to,
        clean_names=args.clean_names,
        match_date=args.match_date,
        api_key=args.api_key,
        api_key_file=args.api_key_file,
    )


def build_kenpom(config: AppConfig) -> KenPom:
    """build the kenpom api config"""
    return KenPom(
        api_key=resolve_api_key(
            config.api_key,
            config.api_key_file,
            KENPOM_API_KEY_ENV,
            "kenpom",
        )
    )


def save_na_counts_by_date(xf: pd.DataFrame, config: AppConfig) -> None:
    """Save per-date NA counts for every column."""
    save_file: str = os.path.join(config.save_to, "all_matches_na_counts_by_date.pqt")
    value_cols: list[str] = [column for column in xf.columns if column != "LABEL_DATE"]
    na_counts: pd.DataFrame = (
        xf[value_cols].isna().groupby(xf["LABEL_DATE"]).sum().reset_index()
    )

    na_counts = na_counts.loc[na_counts[value_cols].sum(axis=1) > 0].sort_values(
        "LABEL_DATE"
    )
    save_parquet(na_counts, save_file)
    update_status(f"saved NA counts by date: {save_file}")


def main() -> None:
    """download and build the raw ncaamb match dataset"""
    global KENPOM
    config: AppConfig = build_app_config(parse_arguments())
    clean_xref: pd.DataFrame
    refdata: pd.DataFrame
    all_matches: pd.DataFrame
    all_names: pd.DataFrame

    KENPOM = build_kenpom(config)

    update_status("getting ref data")
    clean_xref = pd.read_csv(config.clean_names)
    refdata = load_ref_data(clean_xref)

    update_status("getting match data")
    all_matches = build_match_details(refdata, config)
    all_names = build_canonical_names(all_matches, clean_xref, refdata)
    all_matches = merge_refdata(all_matches, all_names, refdata)

    update_status("saving NA counts by date")
    save_na_counts_by_date(all_matches, config)

    update_status("finalizing and saving")
    finalize_and_save(all_matches, config)


if __name__ == "__main__":
    main()

