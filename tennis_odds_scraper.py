from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from utilities import (
    BaseAppConfig,
    add_api_key_args,
    make_save_to_parser,
    resolve_api_key,
    save_csv,
    update_status,
    write_json,
)

__all__: list[str] = []

if __name__ not in {"__main__", "__mp_main__"}:
    raise ImportError(
        "tennis_odds_scraper.py is a script entrypoint and should not be imported"
    )


ODDS_API_KEY_ENV: Final[str] = "ODDS_API"
ODDS_API_BASE_URL: Final[str] = "https://api.the-odds-api.com/v4"
OUTPUT_COLUMNS: Final[list[str]] = [
    "Date",
    "Winner",
    "Loser",
    "Location",
    "Tournament",
    "Court",
    "Surface",
    "Round",
    "Best of",
    "Series",
    "CATEGORY_MATCH_DIVISION",
    "LABEL_SOURCE_EVENT_ID",
    "LABEL_SOURCE_SPORT_KEY",
    "LABEL_COMMENCE_TIME",
    "LABEL_BOOKMAKERS",
]


@dataclass(frozen=True, slots=True)
class AppConfig(BaseAppConfig):
    """hold runtime settings"""

    api_key: str
    sports: tuple[str, ...] = ()
    regions: str = "us"
    markets: str = "h2h"
    odds_format: str = "american"
    timezone_name: str = "America/New_York"
    match_date: date = date.today()
    days_ahead: int = 1
    discover_sports: bool = False
    output_file: str = "games_today.csv"
    raw_file: str = "tennis_odds_raw.json"
    base_url: str = ODDS_API_BASE_URL
    timeout: tuple[float, float] = (5.0, 20.0)

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError(
                "missing odds api key; use --api_key, --api_key_file, "
                f"or set {ODDS_API_KEY_ENV}"
            )
        if self.days_ahead <= 0:
            raise ValueError(f"days_ahead must be positive; received {self.days_ahead}")
        if not all(timeout > 0 for timeout in self.timeout):
            raise ValueError(
                f"timeout values must be positive; received {self.timeout}"
            )
        ZoneInfo(self.timezone_name)
        BaseAppConfig.__post_init__(self)

    @property
    def output_path(self) -> Path:
        """return normalized schedule csv path"""
        return self.output_dir / self.output_file

    @property
    def raw_path(self) -> Path:
        """return raw odds response path"""
        return self.output_dir / self.raw_file


def parse_arguments() -> Namespace:
    """parse cli arguments"""
    parser: ArgumentParser = make_save_to_parser(
        "download bettable tennis matches from the odds api",
        save_to_help="output directory",
    )
    add_api_key_args(parser, "odds api")
    parser.add_argument(
        "--sports",
        nargs="*",
        default=None,
        help="odds api sport keys to query; defaults to active tennis sports",
    )
    parser.add_argument(
        "--discover_sports",
        action="store_true",
        help="query active tennis sport keys even when --sports is provided",
    )
    parser.add_argument("--regions", default="us", help="odds api bookmaker regions")
    parser.add_argument("--markets", default="h2h", help="odds api markets")
    parser.add_argument(
        "--odds_format",
        default="american",
        choices=["american", "decimal"],
        help="odds format",
    )
    parser.add_argument(
        "--timezone",
        dest="timezone_name",
        default="America/New_York",
        help="timezone used for date filtering and output dates",
    )
    parser.add_argument(
        "--match_date",
        type=date.fromisoformat,
        default=date.today(),
        help="first local match date to include in yyyy-mm-dd format",
    )
    parser.add_argument(
        "--days_ahead",
        type=int,
        default=1,
        help="number of local dates to include",
    )
    parser.add_argument(
        "--output_file",
        default="tennis_unplayed_matches.csv",
        help="csv filename for tennis_loader --games",
    )
    parser.add_argument(
        "--raw_file",
        default="tennis_odds_raw.json",
        help="json filename for raw odds api responses",
    )
    return parser.parse_args()


def build_app_config(args: Namespace) -> AppConfig:
    """build runtime config"""
    return AppConfig(
        save_to=args.save_to,
        api_key=resolve_api_key(
            args.api_key,
            args.api_key_file,
            ODDS_API_KEY_ENV,
            "odds api",
        ),
        sports=tuple(args.sports) if args.sports is not None else (),
        regions=args.regions,
        markets=args.markets,
        odds_format=args.odds_format,
        timezone_name=args.timezone_name,
        match_date=args.match_date,
        days_ahead=args.days_ahead,
        discover_sports=args.discover_sports,
        output_file=args.output_file,
        raw_file=args.raw_file,
    )


def odds_get(config: AppConfig, path: str, params: Mapping[str, Any]) -> Any:
    """perform one odds api get request"""
    response: requests.Response = requests.get(
        f"{config.base_url}/{path.lstrip('/')}",
        params={"apiKey": config.api_key, **params},
        timeout=config.timeout,
    )
    if not response.ok:
        raise ValueError(
            f"odds api {path} returned {response.status_code}: "
            f"{response.text[:500]}"
        )
    update_status(
        "odds api quota "
        f"last={response.headers.get('x-requests-last', '?')} "
        f"used={response.headers.get('x-requests-used', '?')} "
        f"remaining={response.headers.get('x-requests-remaining', '?')}"
    )
    return response.json()


def discover_tennis_sports(config: AppConfig) -> tuple[str, ...]:
    """return active tennis sport keys from the odds api"""
    payload: Any = odds_get(config, "sports", {})
    if not isinstance(payload, Sequence):
        raise ValueError(f"expected sports list, received {type(payload)}")
    sports: list[str] = [
        str(item["key"])
        for item in payload
        if isinstance(item, Mapping)
        and item.get("active")
        and item.get("group") == "Tennis"
        and not item.get("has_outrights", False)
    ]
    if not sports:
        raise ValueError("no active non-outright tennis sports returned")
    return tuple(sorted(set(sports)))


def fetch_sport_odds(config: AppConfig, sport_key: str) -> list[Mapping[str, Any]]:
    """fetch h2h odds for one tennis sport"""
    payload: Any = odds_get(
        config,
        f"sports/{sport_key}/odds",
        {
            "regions": config.regions,
            "markets": config.markets,
            "oddsFormat": config.odds_format,
            "dateFormat": "iso",
        },
    )
    if not isinstance(payload, list):
        raise ValueError(f"{sport_key}: expected odds list, received {type(payload)}")
    return [event for event in payload if isinstance(event, Mapping)]


def parse_commence_time(value: str) -> datetime:
    """parse odds api commence time as aware utc datetime"""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def local_match_date(event: Mapping[str, Any], tz: ZoneInfo) -> date:
    """return local match date from one odds event"""
    return parse_commence_time(str(event["commence_time"])).astimezone(tz).date()


def event_has_h2h(event: Mapping[str, Any]) -> bool:
    """return whether event has at least one h2h bookmaker market"""
    bookmakers: Any = event.get("bookmakers", [])
    if not isinstance(bookmakers, Sequence):
        return False
    return any(
        isinstance(bookmaker, Mapping)
        and any(
            isinstance(market, Mapping) and market.get("key") == "h2h"
            for market in bookmaker.get("markets", [])
        )
        for bookmaker in bookmakers
    )


def sport_division(sport_key: str, sport_title: str) -> str:
    """map odds api tennis sport to match division"""
    label: str = f"{sport_key} {sport_title}".upper()
    if "WTA" in label:
        return "WTA"
    if "ATP" in label:
        return "ATP"
    return "UNKNOWN"


def bookmaker_titles(event: Mapping[str, Any]) -> str:
    """return comma-delimited bookmaker titles"""
    bookmakers: Any = event.get("bookmakers", [])
    if not isinstance(bookmakers, Sequence):
        return ""
    titles: list[str] = [
        str(bookmaker.get("title", bookmaker.get("key", "")))
        for bookmaker in bookmakers
        if isinstance(bookmaker, Mapping)
    ]
    return ",".join(title for title in titles if title)


def event_to_row(event: Mapping[str, Any], tz: ZoneInfo) -> dict[str, Any]:
    """convert one odds api event to tennis_loader games csv row"""
    commence_time: datetime = parse_commence_time(str(event["commence_time"]))
    local_time: datetime = commence_time.astimezone(tz)
    sport_key: str = str(event.get("sport_key", ""))
    sport_title: str = str(event.get("sport_title", sport_key))
    tournament: str = sport_title or sport_key
    return {
        "Date": local_time.date().isoformat(),
        "Winner": event["home_team"],
        "Loser": event["away_team"],
        "Location": "",
        "Tournament": tournament,
        "Court": "",
        "Surface": "",
        "Round": "",
        "Best of": "",
        "Series": tournament,
        "CATEGORY_MATCH_DIVISION": sport_division(sport_key, sport_title),
        "LABEL_SOURCE_EVENT_ID": event.get("id", ""),
        "LABEL_SOURCE_SPORT_KEY": sport_key,
        "LABEL_COMMENCE_TIME": local_time.isoformat(),
        "LABEL_BOOKMAKERS": bookmaker_titles(event),
    }


def build_schedule_frame(
    events_by_sport: Mapping[str, Sequence[Mapping[str, Any]]],
    config: AppConfig,
) -> pd.DataFrame:
    """build a tennis_loader-compatible upcoming matches frame"""
    tz: ZoneInfo = ZoneInfo(config.timezone_name)
    end_date: date = config.match_date + timedelta(days=config.days_ahead)
    rows: list[dict[str, Any]] = []
    for events in events_by_sport.values():
        for event in events:
            event_date: date = local_match_date(event, tz)
            if event_date < config.match_date or event_date >= end_date:
                continue
            if not event_has_h2h(event):
                continue
            rows.append(event_to_row(event, tz))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(
        ["Date", "Tournament", "Winner", "Loser"],
        kind="stable",
        ignore_index=True,
    )


def main() -> None:
    """download bettable tennis schedule and save tennis_loader games csv"""
    config: AppConfig = build_app_config(parse_arguments())
    sports: tuple[str, ...] = (
        discover_tennis_sports(config)
        if config.discover_sports or not config.sports
        else config.sports
    )
    update_status(f"fetching odds for sports={sports}")
    events_by_sport: dict[str, list[Mapping[str, Any]]] = {
        sport: fetch_sport_odds(config, sport) for sport in sports
    }
    write_json(
        str(config.raw_path),
        {
            "sports": list(sports),
            "match_date": config.match_date.isoformat(),
            "days_ahead": config.days_ahead,
            "timezone": config.timezone_name,
            "events_by_sport": events_by_sport,
        },
    )
    schedule: pd.DataFrame = build_schedule_frame(events_by_sport, config)
    if schedule.empty:
        raise RuntimeError(
            f"no bettable tennis matches found for {config.match_date} "
            f"through {config.match_date + timedelta(days=config.days_ahead - 1)}"
        )
    save_csv(schedule, config.output_path)
    update_status(f"saved {len(schedule)} matches to {config.output_path}")


if __name__ == "__main__":
    main()
