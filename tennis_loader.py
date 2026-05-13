import io
import os
import re
import unicodedata
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from functools import partial

import numpy as np
import pandas as pd

from utilities import (
    BaseAppConfig,
    add_series_id_columns,
    finalize_grammar_frame,
    make_save_to_parser,
    parallelize_df,
    save_parquet,
    smart_type,
    uppercase_series_columns,
    update_status,
)

TENNIS_URL: str = "http://www.tennis-data.co.uk"
NUMERIC_SET_COLS: list = [
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "Wsets",
    "Lsets",
]


@dataclass(frozen=True, slots=True)
class AppConfig(BaseAppConfig):
    """hold runtime settings"""

    start_year: int = 2007
    games: str | None = None
    files: str | None = None

    def __post_init__(self) -> None:
        BaseAppConfig.__post_init__(self)


TENNIS_FIELDS: list = (
    [
        "Location",
        "Tournament",
        "Date",
        "Court",
        "Surface",
        "Round",
        "Best of",
        "Winner",
        "Loser",
        "WRank",
        "LRank",
        "WPts",
        "LPts",
        "Comment",
        "Series",
    ]
    + NUMERIC_SET_COLS
    + ["CATEGORY_MATCH_DIVISION"]
)

ALTCHARS: dict = {
    "ß": "SS",
    "ẞ": "SS",
    "Æ": "AE",
    "Ǽ": "AE",
    "æ": "AE",
    "ǽ": "AE",
    "Œ": "OE",
    "œ": "OE",
    "Ø": "O",
    "ø": "O",
    "Ł": "L",
    "ł": "L",
    "Đ": "DJ",
    "đ": "DJ",
    "Þ": "TH",
    "þ": "TH",
    "Ð": "D",
    "ð": "D",
}
PREFIX_LIST: list = [
    "AL",
    "DA",
    "DE",
    "DEL",
    "DELLA",
    "DER",
    "DEN",
    "DENN",
    "DI",
    "DOS",
    "DU",
    "EL",
    "LA",
    "LE",
    "LO",
    "MAC",
    "MC",
    "O",
    "SAN",
    "SANTA",
    "ST",
    "VAN",
    "VON",
]


def clean_diacritics(s: str) -> str:
    """clean diacritics"""
    s = "".join(ALTCHARS.get(ch, ch) for ch in s)
    nfkd: str = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def tokenize_name(s: str) -> list:
    """clean tokens"""
    s = clean_diacritics(s).upper()
    s = re.sub(r"[^\w\s\-\.]", " ", s)
    s = s.replace("-", " ")
    s = s.replace(".", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()

    for prefix in [x for x in PREFIX_LIST if s.startswith(f"{x} ")]:
        s = s.replace(f"{prefix} ", prefix, 1)
    return s.split()


def normalize_name(name: str) -> str:
    """normalize player name"""
    toks: list = tokenize_name(name)

    if len(toks) == 1:
        return toks[0]
    initials: list = [x for x in toks[1:] if len(x) == 1]
    idx: str = initials[0] if initials else toks[-1][0]
    return f"{toks[0]} {idx}"


def clean_string(df: pd.DataFrame, replace_dict: dict, fld: str) -> None:
    """normalize string"""
    df[fld] = df[fld].astype(str).str.strip().str.upper()
    df[fld] = df[fld].replace(replace_dict)


def clean_locations(df: pd.DataFrame) -> None:
    """fix raw location names"""
    clean_string(
        df,
        {
            "MARRAKECH": "MARRAKESH",
            "MONREAL": "MONTREAL",
            "QUEENS CLUB": "LONDON",
            "ROGERS CUP": "TORONTO",
            "NUR-SULTAN": "ASTANA",
            "FOREST HILLS": "NEW YORK",
        },
        "CATEGORY_LOCATION",
    )


def clean_tournaments(df: pd.DataFrame) -> None:
    """fix raw tournament names"""
    replace_dict: dict = {
        "APAI INTERNATIONAL": "APIA INTERNATIONAL",
        "COPA CLARO COLSANITAS": "COPA COLSANITAS",
        "COPA COLSANITAS SANTANDER": "COPA COLSANITAS",
        "COPA SONY ERICSSON COLSANITAS": "COPA COLSANITAS",
        "FOREST HILLS SONY ERICSSON WTA TOUR CLASSIC": "FOREST HILLS TENNIS CLASSIC",
        "GARANTI KOZA WTA TOURNAMENT OF CHAMPIONS": "GARANTI KOZA SOFIA OPEN",
        "GENERALI LADIES LINZ OPEN": "LADIES LINZ OPEN",
        "GENERALI LADIES LINZ": "LADIES LINZ OPEN",
        "GERMAN OPEN TENNIS CHAMPIONSHIPS": "GERMAN TENNIS CHAMPIONSHIPS",
        "GUANGZHOU INTERNATIONAL WOMEN'S OPEN": "GUANGZHOU OPEN",
        "GDD-GUANGZHOU INTERNATIONAL WOMENS OPEN": "GUANGZHOU OPEN",
        "INTERNAZIONALI FEMMINILI DI TENNIS DI PALERMO": "INTERNAZIONALI FEMMINILI DI PALERMO",
        "JAPAN WOMEN'S OPEN TENNIS": "JAPAN WOMEN'S TENNIS OPEN",
        "MILLENIUM ESTORIL OPEN": "ESTORIL OPEN",
        "MILLENNIUM ESTORIL OPEN": "ESTORIL OPEN",
        "ABIERTO MEXICANO MIFEL": "ABIERTO MEXICANO",
        "MALAYSIAN OPEN": "MALAYSIA OPEN",
        "REGIONS MORGAN KEEGAN CHAMPIONSHIPS & THE CELLULAR SOUTH CUP": "REGIONS MORGAN KEEGAN CHAMPIONSHIPS",
        "MUTUA MADRILEÑA MADRID OPEN": "MUTUA MADRID OPEN",
        "OPEN GAZ DE FRANCE": "OPEN GDF SUEZ",
        "SHENZHEN LONGGANG GEMDALE OPEN": "SHENZHEN OPEN",
        "THAILAND OPEN 2": "THAILAND OPEN",
        "TORAY PAN PACIFIC OPEN TENNIS TOURNAMENT": "TORAY PAN PACIFIC OPEN",
        "ADELAIDE INTERNATIONAL 1": "ADELAIDE INTERNATIONAL",
        "ADELAIDE INTERNATIONAL 2": "ADELAIDE INTERNATIONAL",
        "NEXT GENERATION ADELAIDE INTERNATIONAL": "ADELAIDE INTERNATIONAL",
        "ATLANTA TENNIS CHAMPIONSHIPS": "ATLANTA OPEN",
        "BB&T ATLANTA OPEN": "ATLANTA OPEN",
        "COMMONWEALTH BANK TOURNAMENT OF CHAMPIONS": "COMMONWEALTH BANK TENNIS CLASSIC",
        "DAVIDOFF SWISS INDOORS": "SWISS INDOORS",
        "CATELLA SWEDISH OPEN": "SWEDISH OPEN",
        "COLLECTOR SWEDISH OPEN": "SWEDISH OPEN",
        "ERICSSON OPEN": "SWEDISH OPEN",
        "NORDEA OPEN": "SWEDISH OPEN",
        "SKISTAR SWEDISH OPEN": "SWEDISH OPEN",
        "SONY SWEDISH OPEN": "SWEDISH OPEN",
        "SERBIA LADIES OPEN": "BELGRADE OPEN",
        "SERBIA OPEN": "BELGRADE OPEN",
        "TORONTO": "ROGERS CUP",
    }
    clean_string(df, replace_dict, "CATEGORY_TOURNAMENT")


def get_file(items: tuple) -> pd.DataFrame:
    """read file and apply division"""
    division, filename = items
    try:
        rf: pd.DataFrame = pd.read_excel(filename, engine="calamine")
        rf["CATEGORY_MATCH_DIVISION"] = division
        return rf
    except Exception as ex:
        update_status(f"cant get {filename} for {division} b/c {ex}")
        return pd.DataFrame()


def get_match_files(this_year, file_path) -> pd.DataFrame:
    """fetch tennis results from files"""

    def adj_ftype(filename) -> str:
        """helper to set correct extension"""
        return filename if os.path.isfile(filename) else filename.replace("xlsx", "xls")

    endpoints: dict = {
        "ATP": adj_ftype(f"{file_path}/men/{this_year}.xlsx"),
        "WTA": adj_ftype(f"{file_path}/women/{this_year}.xlsx"),
    }
    return pd.concat(
        [get_file(items) for items in endpoints.items()], ignore_index=True
    ).sort_values("Date")


def get_match_urls(this_year: int) -> pd.DataFrame:
    """fetch yearly tennis data from url"""

    endpoints: dict = {
        "ATP": f"{TENNIS_URL}/{this_year}/{this_year}.xlsx",
        "WTA": f"{TENNIS_URL}/{this_year}w/{this_year}.xlsx",
    }
    dataset: list = [get_file(items) for items in endpoints.items()]
    if dataset:
        df: pd.DataFrame = pd.concat(dataset, ignore_index=True).reset_index(drop=True)
        return df if df.empty else df.sort_values("Date")
    else:
        return pd.DataFrame()


def get_hist_matches(years: list, file_path: str | None) -> pd.DataFrame:
    """parallelize getting tennis data files"""

    df: pd.DataFrame = parallelize_df(
        partial(get_match_files, file_path=file_path) if file_path else get_match_urls,
        years,
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["IS_DATE_NA"] = df["Date"].isna()
    df["Date"] = df.groupby(["Location", "Tournament"], dropna=False)["Date"].ffill()
    df.loc[df["IS_DATE_NA"], "Date"] = df.loc[df["IS_DATE_NA"], "Date"].apply(
        lambda value: value + pd.Timedelta(days=1)
    )
    df[[c for c in TENNIS_FIELDS if c not in df.columns]] = np.nan
    return df[TENNIS_FIELDS]


def clean_rounds(df: pd.DataFrame) -> None:
    """normalize round labels"""
    replace_dict: dict = {
        "FIRST ROUND": "1ST ROUND",
        "SECOND ROUND": "2ND ROUND",
        "THIRD ROUND": "3RD ROUND",
        "FOURTH ROUND": "4TH ROUND",
        "THIRD PLACE": "3RD PLACE",
        "THE FINAL": "FINAL",
        "FINALS": "FINAL",
    }
    clean_string(df, replace_dict, "CATEGORY_ROUND")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """uppercase cols, fix types, and rename to the engine grammar"""

    for col in [x for x in df.columns if df[x].dtype == "object"]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    for col in [x for x in NUMERIC_SET_COLS if x in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.columns = df.columns.str.upper().str.replace(" ", "_")

    for col in [x for x in ["WINNER", "LOSER"] if x in df.columns]:
        df[col] = df[col].map(normalize_name)

    return df.rename(
        columns={
            "DATE": "LABEL_DATE",
            "WINNER": "SERIES_WINNER",
            "LOSER": "SERIES_LOSER",
            "WRANK": "FEATURE_WINNER_RANK",
            "LRANK": "FEATURE_LOSER_RANK",
            "WPTS": "FEATURE_WINNER_POINTS",
            "LPTS": "FEATURE_LOSER_POINTS",
            "W1": "FEATURE_WINNER_SET_1",
            "W2": "FEATURE_WINNER_SET_2",
            "W3": "FEATURE_WINNER_SET_3",
            "W4": "FEATURE_WINNER_SET_4",
            "W5": "FEATURE_WINNER_SET_5",
            "L1": "FEATURE_LOSER_SET_1",
            "L2": "FEATURE_LOSER_SET_2",
            "L3": "FEATURE_LOSER_SET_3",
            "L4": "FEATURE_LOSER_SET_4",
            "L5": "FEATURE_LOSER_SET_5",
            "WSETS": "FEATURE_WINNER_SETS",
            "LSETS": "FEATURE_LOSER_SETS",
            "LOCATION": "CATEGORY_LOCATION",
            "TOURNAMENT": "CATEGORY_TOURNAMENT",
            "COURT": "CATEGORY_COURT",
            "SURFACE": "CATEGORY_SURFACE",
            "ROUND": "CATEGORY_ROUND",
            "BEST_OF": "CATEGORY_SETS",
            "SERIES": "CATEGORY_SERIES",
            "COMMENT": "CATEGORY_COMMENT",
        }
    )


def get_player_data(match_frame: pd.DataFrame) -> pd.DataFrame:
    """player legend of id and last rank and points"""

    widx: pd.Series = match_frame.groupby("SERIES_WINNER")["LABEL_DATE"].idxmax()
    w: pd.DataFrame = match_frame.loc[
        widx,
        [
            "SERIES_WINNER",
            "FEATURE_WINNER_RANK",
            "FEATURE_WINNER_POINTS",
            "LABEL_DATE",
        ],
    ].rename(
        columns={
            "SERIES_WINNER": "PLAYER",
            "FEATURE_WINNER_RANK": "PRANK",
            "FEATURE_WINNER_POINTS": "PPTS",
        }
    )

    lidx: pd.Series = match_frame.groupby("SERIES_LOSER")["LABEL_DATE"].idxmax()
    l: pd.DataFrame = match_frame.loc[
        lidx,
        [
            "SERIES_LOSER",
            "FEATURE_LOSER_RANK",
            "FEATURE_LOSER_POINTS",
            "LABEL_DATE",
        ],
    ].rename(
        columns={
            "SERIES_LOSER": "PLAYER",
            "FEATURE_LOSER_RANK": "PRANK",
            "FEATURE_LOSER_POINTS": "PPTS",
        }
    )

    ids: pd.DataFrame = (
        pd.concat([w, l], ignore_index=True)
        .sort_values("LABEL_DATE")
        .drop_duplicates(subset="PLAYER", keep="last")
        .reset_index(drop=True)
    )
    return ids[["PLAYER", "PRANK", "PPTS"]]


def handle_missing_players(
    df: pd.DataFrame, missing_keys: list, hist_universe: list
) -> pd.DataFrame:
    """handle missing players from the command line"""
    anchors: list = [
        x
        for x in hist_universe
        if any(y in x for y in [z[: z.find(" ")] for z in missing_keys])
    ]
    update_status(f"MISSING PLAYERS: {sorted(missing_keys)}")
    update_status(f"possible alts: {sorted(anchors)}")
    for key in missing_keys:
        rows_mask: np.ndarray = (df["WINNER"] == key) | (df["LOSER"] == key)
        while True:
            choice: str = (
                input(f"\nAction for {key}: (E)dit / (D)elete rows / (S)kip [E/D/S]: ")
                .strip()
                .lower()
            )
            if choice in {"e", "d", "s"}:
                break

        if choice == "d":
            df = df.loc[~rows_mask].copy()
            update_status(f"Dropped {rows_mask.sum()} row(s) for {key}.")
            continue

        if choice == "s":
            update_status(f"Keeping {key} unresolved (IDs/ranks will be NaN).")
            continue

        new_key: str = input("Type replacement:").strip().upper()
        df.loc[df["WINNER"] == key, "WINNER"] = new_key
        df.loc[df["LOSER"] == key, "LOSER"] = new_key
        update_status(f"Replaced '{key}' -> '{new_key}' in today's rows.")

    return df


def get_todays_matches(
    match_frame: pd.DataFrame,
    input_file: str | None,
) -> pd.DataFrame:
    """merge a CSV of today's matches into historical frame"""
    if not input_file:
        return match_frame

    df: pd.DataFrame = pd.read_csv(input_file)
    df.columns = df.columns.str.strip().str.upper()
    if "LABEL_DATE" in df.columns:
        df = df.rename(columns={"LABEL_DATE": "DATE"})
    df["COMMENT"] = "INCOMPLETE"
    df = normalize_columns(df)
    missing_cols: list = [c for c in match_frame.columns if c not in df.columns]
    df[missing_cols] = np.nan

    return pd.concat([match_frame, df[match_frame.columns]], ignore_index=True)


def parse_arguments() -> Namespace:
    """arg parser"""
    parser: ArgumentParser = make_save_to_parser(
        "download & normalize tennis data (tennis-data.co.uk)",
        save_to_help="output directory",
    )
    parser.add_argument(
        "-y", "--start_year", type=int, default=2007, help="start year (inclusive)"
    )
    parser.add_argument(
        "-g", "--games", type=str, help="CSV of today's matches to append/validate"
    )
    parser.add_argument("-f", "--files", type=str, help="path of files")
    return parser.parse_args()


def build_app_config(args: Namespace) -> AppConfig:
    """build the app config"""
    return AppConfig(
        save_to=args.save_to,
        start_year=args.start_year,
        games=args.games,
        files=args.files,
    )


def clean_sets(df: pd.DataFrame) -> pd.DataFrame:
    """normalize set counts"""
    max_sets: pd.DataFrame = (
        df.groupby(["CATEGORY_TOURNAMENT", "CATEGORY_MATCH_DIVISION"])
        .agg({"FEATURE_WINNER_SETS": "max", "FEATURE_LOSER_SETS": "max"})
        .reset_index()
    )
    max_sets["MAX_SETS"] = max_sets[
        ["FEATURE_WINNER_SETS", "FEATURE_LOSER_SETS"]
    ].max(axis=1)
    df = df.merge(
        max_sets[["CATEGORY_TOURNAMENT", "CATEGORY_MATCH_DIVISION", "MAX_SETS"]],
        on=["CATEGORY_TOURNAMENT", "CATEGORY_MATCH_DIVISION"],
    )
    df["CATEGORY_SETS"] = df["CATEGORY_SETS"].fillna(df["MAX_SETS"])
    df["CATEGORY_SETS"] = np.where(
        (df["FEATURE_WINNER_SETS"] <= 2) & (df["CATEGORY_COMMENT"] == "COMPLETED"),
        3,
        df["CATEGORY_SETS"],
    )
    df = df.drop(columns="MAX_SETS")
    return df


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """normalize categorical values"""
    clean_tournaments(df)
    clean_locations(df)
    clean_rounds(df)
    df["CATEGORY_SURFACE"] = df["CATEGORY_SURFACE"].replace({"GREENSET": "HARD"})
    df["CATEGORY_SERIES"] = df["CATEGORY_SERIES"].replace({0: "OTHER", "NAN": "OTHER"})
    df["CATEGORY_COMMENT"] = df["CATEGORY_COMMENT"].replace(
        {"RRTIRED": "RETIRED", "WALKOER": "WALKOVER"}
    )
    df = uppercase_series_columns(df)
    df = smart_type(df)
    df = add_series_id_columns(df)
    labels: list = sorted([x for x in df.columns if x.startswith("LABEL")])
    categories: list = sorted([x for x in df.columns if x.startswith("CATEGORY")])
    series_cols: list = sorted([x for x in df.columns if x.startswith("SERIES")])
    feature_cols: list = sorted([x for x in df.columns if x.startswith("FEATURE")])
    is_cols: list = sorted([x for x in df.columns if x.startswith("IS")])
    return finalize_grammar_frame(
        df[labels + series_cols + feature_cols + is_cols + categories]
    )


def main() -> None:
    """download and normalize tennis data"""
    config: AppConfig = build_app_config(parse_arguments())

    years: list = list(range(config.start_year, date.today().year + 1))
    update_status(f"fetching {len(years)} seasons")
    match_frame: pd.DataFrame = get_hist_matches(years, config.files)
    match_frame = normalize_columns(match_frame)

    update_status(
        f"hist matches with shape {match_frame.shape} and last date: {match_frame['LABEL_DATE'].max()}"
    )
    # match_frame["CATEGORY_SERIES"] = match_frame["CATEGORY_SERIES"].fillna("UNKNOWN")

    match_frame = get_todays_matches(match_frame, config.games)
    # match_frame = normalize_columns(match_frame)
    match_frame = clean_frame(match_frame)
    match_frame["CATEGORY_SERIES"] = (
        match_frame["CATEGORY_SERIES"].fillna("UNKNOWN").str.upper()
    )

    save_file: str = os.path.join(config.save_to, "tennis_all_matches.pqt")
    update_status(f"saving: {save_file}, shape: {match_frame.shape}")
    for this_col in [
        x
        for x in match_frame.columns
        if x.startswith("CATEGORY") and x != "CATEGORY_TOURNAMENT"
    ]:
        try:
            update_status(str(sorted(match_frame[this_col].unique())))
        except Exception:
            update_status(f"cant show uniques for {this_col}")

    info_buffer: io.StringIO = io.StringIO()
    match_frame.info(buf=info_buffer, verbose=True)
    save_parquet(match_frame, save_file)
    update_status(info_buffer.getvalue())


if __name__ == "__main__":
    main()
