from __future__ import annotations

import io
import re
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from utilities import (
    BaseAppConfig,
    JSONRequest,
    add_api_key_args,
    add_series_id_columns,
    fetch_json_retries,
    finalize_grammar_frame,
    load_json,
    make_save_to_parser,
    resolve_api_key,
    save_parquet,
    smart_type,
    uppercase_series_columns,
    update_status,
    write_json,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATCHSTAT_API_KEY_ENV: Final[str] = "MATCHSTAT_API_KEY"
MATCHSTAT_HOST: Final[str] = "tennis-api-atp-wta-itf.p.rapidapi.com"
MATCHSTAT_BASE_URL: Final[str] = f"https://{MATCHSTAT_HOST}/tennis/v2"

DIVISIONS: Final[tuple[str, str]] = ("atp", "wta")
PAGE_SIZE: Final[int] = 50
REQUEST_SPACING: Final[float] = 0.65  # ~92 req/min, under 100/min cap

# included fields when fetching upcoming fixtures
FIXTURE_INCLUDE: Final[str] = (
    "round,tournament,tournament.court"
    ",tournament.rank,tournament.country,h2h"
)

# included fields when fetching historical tournament results
RESULTS_INCLUDE: Final[str] = "round"

TOURNAMENT_CACHE_FILE: Final[str] = "tournament_info_cache.json"

# tiers excluded when include_futures is False
_SKIP_TIERS: Final[frozenset[str]] = frozenset({"FUTURE", "FUTURES"})

_INDOOR_MARKERS: Final[frozenset[str]] = frozenset(
    {"INDOOR HARD", "INDOOR CLAY", "CARPET"}
)
_SURFACE_MAP: Final[dict[str, str]] = {
    "INDOOR HARD": "Hard",
    "INDOOR CLAY": "Clay",
}

_SET_RE: re.Pattern = re.compile(r"(\d+)-(\d+)(?:\(\d+\))?")
_RETIRED_RE: re.Pattern = re.compile(r"\bRET(?:IRED)?\b", re.IGNORECASE)
_WALKOVER_RE: re.Pattern = re.compile(
    r"\bW[/ ]?O\b|\bWALKOVER\b", re.IGNORECASE
)
_NONDIGIT_RE: re.Pattern = re.compile(r"[^\d.]")

# fallback round-name mapping for when include=round is unavailable
_ROUND_NAMES: Final[dict[int, str]] = {
    1: "Q1", 2: "Q2", 3: "Q3",
    4: "R128", 5: "R64", 6: "R32", 7: "R16",
    8: "Quarterfinals", 9: "Semifinals", 10: "Final",
    11: "3rd Place", 12: "Final",
}

# module-level rate-limit clock shared across all calls in this process
_next_call_at: list[float] = [0.0]

TournamentMeta = dict[str, Any]  # keys: site, surface, prize, draw_size
TournamentCache = dict[str, TournamentMeta]  # key: "division/season_id"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AppConfig(BaseAppConfig):
    """hold runtime settings"""

    api_key: str
    start_year: int = 2007
    include_futures: bool = False
    tournament_info: bool = True
    timeout: tuple[float, float] = (5.0, 30.0)
    output_file: str = "tennis_all_matches.pqt"

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError(
                "missing matchstat api key; use --api_key, "
                f"--api_key_file, or set {MATCHSTAT_API_KEY_ENV} env var"
            )
        BaseAppConfig.__post_init__(self)

    @property
    def output_path(self) -> Path:
        """return the output parquet path"""
        return self.output_dir / self.output_file

    @property
    def tournament_cache_path(self) -> Path:
        """return the tournament metadata cache path"""
        return self.output_dir / TOURNAMENT_CACHE_FILE


# ---------------------------------------------------------------------------
# API layer
# ---------------------------------------------------------------------------


def _api_get(
    config: AppConfig, path: str, params: dict[str, Any]
) -> Any:
    """one rate-limited matchstat API call"""
    wait = _next_call_at[0] - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _next_call_at[0] = time.monotonic() + REQUEST_SPACING

    return fetch_json_retries(
        JSONRequest(
            url=f"{MATCHSTAT_BASE_URL}/{path.lstrip('/')}",
            params=params,
            headers={
                "x-rapidapi-key": config.api_key,
                "x-rapidapi-host": MATCHSTAT_HOST,
            },
            timeout=config.timeout,
        )
    )


def _has_next_page(response: Any, fixtures: list[Any]) -> bool:
    """detect whether the API signals a next page"""
    if isinstance(response, dict) and "hasNextPage" in response:
        return bool(response["hasNextPage"])
    last = fixtures[-1] if fixtures else None
    if isinstance(last, dict) and "hasNextPage" in last:
        return bool(last["hasNextPage"])
    return len(fixtures) >= PAGE_SIZE


def _extract_fixtures(response: Any) -> list[dict[str, Any]]:
    """pull the fixture list from whatever shape the response takes"""
    if isinstance(response, list):
        return [f for f in response if isinstance(f, dict)]
    if isinstance(response, dict):
        for key in ("data", "fixtures", "results"):
            if isinstance(response.get(key), list):
                return [f for f in response[key] if isinstance(f, dict)]
    return []


def fetch_all_pages(
    config: AppConfig,
    division: str,
    path_suffix: str,
) -> list[dict[str, Any]]:
    """page through all upcoming fixtures for one division endpoint"""
    all_fixtures: list[dict[str, Any]] = []
    page_no = 1
    while True:
        response = _api_get(
            config,
            f"{division}/fixtures/{path_suffix}",
            {
                "include": FIXTURE_INCLUDE,
                "filter": "PlayerGroup:singles",
                "pageSize": PAGE_SIZE,
                "pageNo": page_no,
            },
        )
        if response is None:
            update_status(
                f"null response: {division} {path_suffix} page {page_no}"
            )
            break
        if page_no == 1 and not _extract_fixtures(response):
            update_status(
                f"unexpected response shape: {str(response)[:300]}"
            )
        fixtures = _extract_fixtures(response)
        all_fixtures.extend(fixtures)
        update_status(
            f"{division} {path_suffix} p{page_no}: "
            f"{len(fixtures)} fixtures (total: {len(all_fixtures)})"
        )
        if not _has_next_page(response, fixtures):
            break
        page_no += 1
    return all_fixtures


# ---------------------------------------------------------------------------
# Tournament info cache (prize money / draw size for upcoming)
# ---------------------------------------------------------------------------


def _load_tournament_cache(cache_path: Path) -> TournamentCache:
    """load persisted cache from disk; returns empty dict if missing"""
    if not cache_path.exists():
        return {}
    try:
        return load_json(str(cache_path))
    except Exception as ex:
        update_status(f"tournament cache unreadable ({ex}), resetting")
        return {}


def _save_tournament_cache(
    cache: TournamentCache, cache_path: Path
) -> None:
    """save the tournament metadata cache"""
    write_json(str(cache_path), cache)


def _fetch_tournament_info(
    config: AppConfig,
    division: str,
    season_id: int,
    cache: TournamentCache,
    cache_path: Path,
) -> TournamentMeta:
    """fetch and cache one tournament season's metadata"""
    key = f"{division}/{season_id}"
    if key in cache:
        return cache[key]

    response = _api_get(
        config, f"{division}/tournament/info/{season_id}", {}
    )
    data: dict = response if isinstance(response, dict) else {}

    raw_prize = data.get("prize", data.get("singlesPrize", ""))
    try:
        prize: float = (
            float(_NONDIGIT_RE.sub("", str(raw_prize)))
            if raw_prize
            else np.nan
        )
    except ValueError:
        prize = np.nan

    raw_draw = data.get(
        "draw_size", data.get("drawSize", data.get("draw", ""))
    )
    try:
        draw_size: float = float(str(raw_draw)) if raw_draw else np.nan
    except ValueError:
        draw_size = np.nan

    # API has a typo: "coutry" not "country"
    coutry: dict = data.get("coutry") or data.get("country") or {}
    country_name: str = str(coutry.get("name", "") or "").strip()
    court: dict = data.get("court") or {}
    surface: str = str(court.get("name", "") or "").strip()

    meta: TournamentMeta = {
        "site": country_name,
        "surface": surface,
        "prize": prize,
        "draw_size": draw_size,
    }
    cache[key] = meta
    _save_tournament_cache(cache, cache_path)
    return meta


def build_tournament_cache(
    config: AppConfig,
    upcoming_by_division: dict[str, list[dict[str, Any]]],
) -> TournamentCache:
    """fetch info for every unique (division, tournament_id) in upcoming"""
    cache: TournamentCache = _load_tournament_cache(
        config.tournament_cache_path
    )

    seen: set[tuple[str, int]] = {
        (div, int(f["tournament"]["id"]))
        for div, fixtures in upcoming_by_division.items()
        for f in fixtures
        if isinstance(f.get("tournament"), dict)
        and f["tournament"].get("id")
    }
    already: set[tuple[str, int]] = {
        (k.split("/")[0], int(k.split("/")[1])) for k in cache
    }
    pairs: list[tuple[str, int]] = sorted(seen - already)

    if pairs:
        update_status(
            f"fetching tournament info for {len(pairs)} new seasons"
        )
    for div, season_id in pairs:
        _fetch_tournament_info(
            config, div, season_id, cache, config.tournament_cache_path
        )

    return cache


# ---------------------------------------------------------------------------
# Shared row-building helpers
# ---------------------------------------------------------------------------


def _court_surface(court_name: str) -> tuple[str, str]:
    """return (Court, Surface) from matchstat court name"""
    upper = (court_name or "").strip().upper()
    is_indoor = upper in _INDOOR_MARKERS or "INDOOR" in upper
    court = "Indoor" if is_indoor else "Outdoor"
    surface = _SURFACE_MAP.get(upper, court_name.strip()) or ""
    return court, surface


def _best_of(rank_name: str, division: str) -> int:
    """only ATP Grand Slams are best of 5"""
    return (
        5
        if "GRAND SLAM" in rank_name.upper() and division == "atp"
        else 3
    )


def _parse_seed(raw: Any) -> float:
    """numeric seeds only; wc, q, ll, and pr become nan"""
    if raw is None or str(raw).strip() == "":
        return np.nan
    try:
        return float(str(raw).strip())
    except ValueError:
        return np.nan


def _parse_result(result: str) -> dict[str, Any]:
    """parse score string into per-set scores and match comment"""
    row: dict[str, Any] = {
        f"FEATURE_WINNER_SET_{i}": np.nan for i in range(1, 6)
    }
    row.update({f"FEATURE_LOSER_SET_{i}": np.nan for i in range(1, 6)})
    row["FEATURE_WINNER_SETS"] = np.nan
    row["FEATURE_LOSER_SETS"] = np.nan

    result = (result or "").strip()
    if not result:
        row["CATEGORY_COMMENT"] = "INCOMPLETE"
        return row

    if _WALKOVER_RE.search(result):
        row["CATEGORY_COMMENT"] = "WALKOVER"
    elif _RETIRED_RE.search(result):
        row["CATEGORY_COMMENT"] = "RETIRED"
    else:
        row["CATEGORY_COMMENT"] = "COMPLETED"

    sets = _SET_RE.findall(result)
    w_wins = l_wins = 0
    for i, (ws, ls) in enumerate(sets[:5], start=1):
        row[f"FEATURE_WINNER_SET_{i}"] = int(ws)
        row[f"FEATURE_LOSER_SET_{i}"] = int(ls)
        if int(ws) > int(ls):
            w_wins += 1
        else:
            l_wins += 1
    if sets:
        row["FEATURE_WINNER_SETS"] = w_wins
        row["FEATURE_LOSER_SETS"] = l_wins

    return row


# ---------------------------------------------------------------------------
# Upcoming fixture → row
# ---------------------------------------------------------------------------


def fixture_to_row(
    fixture: dict[str, Any],
    division: str,
    tournament_meta: TournamentMeta | None = None,
    is_upcoming: bool = False,
) -> dict[str, Any]:
    """map one upcoming API fixture to a grammar row"""
    tournament: dict = fixture.get("tournament") or {}
    rank: dict = tournament.get("rank") or {}
    court: dict = tournament.get("court") or {}
    p1: dict = fixture.get("player1") or {}
    p2: dict = fixture.get("player2") or {}
    rnd: dict = fixture.get("round") or {}
    h2h: dict = fixture.get("h2h") or {}
    meta: TournamentMeta = tournament_meta or {}

    raw_surface = court.get("name", "") or meta.get("surface", "")
    court_val, surface_val = _court_surface(raw_surface)
    rank_name: str = rank.get("name", "")

    location: str = meta.get("site") or tournament.get("countryAcr", "")

    # H2H is career-total and would leak future results into historical
    # training rows; only populate it for upcoming matches.
    h2h_p1 = (
        float(h2h["player1AllWins"]) if is_upcoming and h2h else np.nan
    )
    h2h_p2 = (
        float(h2h["player2AllWins"]) if is_upcoming and h2h else np.nan
    )

    def _up(s: Any) -> str:
        return str(s or "").strip().upper()

    row: dict[str, Any] = {
        "LABEL_DATE": (fixture.get("date") or "")[:10],
        "LABEL_SOURCE_FIXTURE_ID": fixture.get("id", ""),
        "LABEL_COMMENCE_TIME": (fixture.get("date") or ""),
        "SERIES_WINNER": _up(p1.get("name")),
        "SERIES_LOSER": _up(p2.get("name")),
        "FEATURE_WINNER_RANK": np.nan,
        "FEATURE_LOSER_RANK": np.nan,
        "FEATURE_WINNER_POINTS": np.nan,
        "FEATURE_LOSER_POINTS": np.nan,
        "FEATURE_P1_SEED": _parse_seed(fixture.get("seed1")),
        "FEATURE_P2_SEED": _parse_seed(fixture.get("seed2")),
        "FEATURE_P1_H2H_WINS": h2h_p1,
        "FEATURE_P2_H2H_WINS": h2h_p2,
        "FEATURE_DRAW_SIZE": meta.get("draw_size", np.nan),
        "FEATURE_PRIZE_MONEY": meta.get("prize", np.nan),
        "CATEGORY_LOCATION": _up(location),
        "CATEGORY_TOURNAMENT": _up(tournament.get("name")),
        "CATEGORY_COURT": _up(court_val),
        "CATEGORY_SURFACE": _up(surface_val),
        "CATEGORY_ROUND": _up(rnd.get("name")),
        "CATEGORY_SETS": _best_of(rank_name, division),
        "CATEGORY_SERIES": _up(rank_name),
        "CATEGORY_MATCH_DIVISION": division.upper(),
        "CATEGORY_P1_COUNTRY": _up(p1.get("countryAcr")),
        "CATEGORY_P2_COUNTRY": _up(p2.get("countryAcr")),
    }
    row.update(_parse_result(fixture.get("result", "")))
    return row


def build_frame(
    fixtures: list[dict[str, Any]],
    division: str,
    tournament_cache: TournamentCache,
    is_upcoming: bool,
) -> pd.DataFrame:
    """convert upcoming fixtures to a DataFrame"""
    rows = [
        fixture_to_row(
            f,
            division,
            tournament_meta=tournament_cache.get(
                f"{division}/{f.get('tournament', {}).get('id', '')}"
            ),
            is_upcoming=is_upcoming,
        )
        for f in fixtures
        if f
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Historical tournament results → row
# ---------------------------------------------------------------------------


def _is_main_tour(entry: dict[str, Any]) -> bool:
    """return True for ATP/WTA main tour + challengers; exclude ITF futures"""
    tier = str(entry.get("tier") or "").strip().upper()
    rank_id = int(entry.get("rankId") or 0)
    if any(skip in tier for skip in _SKIP_TIERS):
        return False
    return bool(tier) or rank_id > 0


def fetch_calendar(
    config: AppConfig,
    division: str,
    year: int,
) -> list[dict[str, Any]]:
    """return all tournament season entries for one division and year"""
    response = _api_get(
        config,
        f"{division}/tournament/calendar/{year}",
        {},
    )
    if isinstance(response, list):
        return [e for e in response if isinstance(e, dict)]
    if isinstance(response, dict):
        for key in ("data", "tournaments", "calendar"):
            val = response.get(key)
            if isinstance(val, list):
                return [e for e in val if isinstance(e, dict)]
    return []


def fetch_tournament_results(
    config: AppConfig,
    division: str,
    season_id: int,
) -> list[dict[str, Any]]:
    """return singles match results for one tournament season"""
    response = _api_get(
        config,
        f"{division}/tournament/results/{season_id}",
        {"include": RESULTS_INCLUDE},
    )
    if not isinstance(response, dict):
        return []
    data = response.get("data")
    if isinstance(data, dict):
        singles = data.get("singles") or []
    elif isinstance(data, list):
        singles = data
    else:
        singles = []
    return [m for m in singles if isinstance(m, dict)]


def result_to_row(
    match: dict[str, Any],
    division: str,
    calendar_entry: dict[str, Any],
) -> dict[str, Any] | None:
    """convert a tournament results match to a grammar row; None if unknown winner"""
    p1: dict = match.get("player1") or {}
    p2: dict = match.get("player2") or {}
    p1_id = match.get("player1Id")
    p2_id = match.get("player2Id")
    winner_id = match.get("match_winner")

    if winner_id is None:
        return None
    if winner_id == p1_id or winner_id == p1.get("id"):
        winner, loser = p1, p2
    elif winner_id == p2_id or winner_id == p2.get("id"):
        winner, loser = p2, p1
    else:
        return None

    # prefer embedded round object (include=round); fall back to roundId map
    rnd: dict = match.get("round") or {}
    round_id = int(match.get("roundId") or 0)
    round_name = (
        rnd.get("name") or _ROUND_NAMES.get(round_id, f"Round_{round_id}")
    )

    # tournament metadata comes directly from the calendar entry
    court_obj: dict = calendar_entry.get("court") or {}
    coutry: dict = (
        calendar_entry.get("coutry") or calendar_entry.get("country") or {}
    )
    tier = str(calendar_entry.get("tier") or "")
    rank_name = tier

    raw_surface = court_obj.get("name", "")
    court_val, surface_val = _court_surface(raw_surface)
    location = str(coutry.get("name", "") or "").strip()

    raw_draw = calendar_entry.get("draw_size")
    try:
        draw_size: float = float(str(raw_draw)) if raw_draw else np.nan
    except ValueError:
        draw_size = np.nan

    def _up(s: Any) -> str:
        return str(s or "").strip().upper()

    row: dict[str, Any] = {
        "LABEL_DATE": (match.get("date") or "")[:10],
        "LABEL_SOURCE_FIXTURE_ID": match.get("id", ""),
        "LABEL_COMMENCE_TIME": (match.get("date") or ""),
        "SERIES_WINNER": _up(winner.get("name")),
        "SERIES_LOSER": _up(loser.get("name")),
        "FEATURE_WINNER_RANK": np.nan,
        "FEATURE_LOSER_RANK": np.nan,
        "FEATURE_WINNER_POINTS": np.nan,
        "FEATURE_LOSER_POINTS": np.nan,
        "FEATURE_P1_SEED": np.nan,
        "FEATURE_P2_SEED": np.nan,
        "FEATURE_P1_H2H_WINS": np.nan,
        "FEATURE_P2_H2H_WINS": np.nan,
        "FEATURE_DRAW_SIZE": draw_size,
        "FEATURE_PRIZE_MONEY": np.nan,
        "CATEGORY_LOCATION": _up(location),
        "CATEGORY_TOURNAMENT": _up(calendar_entry.get("name")),
        "CATEGORY_COURT": _up(court_val),
        "CATEGORY_SURFACE": _up(surface_val),
        "CATEGORY_ROUND": _up(round_name),
        "CATEGORY_SETS": _best_of(rank_name, division),
        "CATEGORY_SERIES": _up(rank_name),
        "CATEGORY_MATCH_DIVISION": division.upper(),
        "CATEGORY_P1_COUNTRY": _up(winner.get("countryAcr")),
        "CATEGORY_P2_COUNTRY": _up(loser.get("countryAcr")),
    }
    row.update(_parse_result(match.get("result", "")))
    return row


def build_hist_frame(
    matches: list[dict[str, Any]],
    division: str,
) -> pd.DataFrame:
    """convert tournament results matches to a DataFrame"""
    rows = []
    for match in matches:
        calendar_entry = match.get("_calendar_entry") or {}
        row = result_to_row(match, division, calendar_entry)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Fetch orchestration
# ---------------------------------------------------------------------------


def fetch_historical(
    config: AppConfig,
) -> dict[str, list[dict[str, Any]]]:
    """fetch completed matches via calendar + tournament/results"""
    today = date.today()
    by_division: dict[str, list[dict[str, Any]]] = {d: [] for d in DIVISIONS}

    for year in range(config.start_year, today.year + 1):
        for division in DIVISIONS:
            update_status(f"fetching {division} calendar {year}")
            calendar = fetch_calendar(config, division, year)

            tournaments = (
                [e for e in calendar if _is_main_tour(e)]
                if not config.include_futures
                else calendar
            )
            update_status(
                f"{division} {year}: {len(tournaments)} tournaments "
                f"(of {len(calendar)} total)"
            )

            for entry in tournaments:
                season_id = entry.get("id")
                if not season_id:
                    continue
                results = fetch_tournament_results(
                    config, division, int(season_id)
                )
                if results:
                    update_status(
                        f"  {division} {entry.get('name', season_id)}: "
                        f"{len(results)} results"
                    )
                    for match in results:
                        match["_calendar_entry"] = entry
                    by_division[division].extend(results)

    return by_division


def fetch_upcoming(
    config: AppConfig,
) -> dict[str, list[dict[str, Any]]]:
    """fetch today's unplayed matches"""
    today = date.today().isoformat()
    update_status(f"fetching upcoming {today}")
    by_division: dict[str, list[dict[str, Any]]] = {}
    for division in DIVISIONS:
        raw = fetch_all_pages(config, division, today)
        by_division[division] = [
            f for f in raw if f.get("complete") is None
        ]
    return by_division


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> Namespace:
    """parse cli arguments"""
    parser: ArgumentParser = make_save_to_parser(
        "download tennis data from matchstat "
        "(tennis-api-atp-wta-itf rapidapi)",
        save_to_help="output directory",
    )
    add_api_key_args(parser, "matchstat rapidapi")
    parser.add_argument(
        "-y",
        "--start_year",
        type=int,
        default=2007,
        help="first historical year to fetch (default 2007)",
    )
    parser.add_argument(
        "--include_futures",
        action="store_true",
        help="include ITF futures tournaments in historical fetch",
    )
    parser.add_argument(
        "--no_tournament_info",
        action="store_true",
        help="skip prize/draw_size tournament info calls for upcoming",
    )
    parser.add_argument(
        "--output_file",
        default="tennis_all_matches.pqt",
        help="output parquet filename",
    )
    return parser.parse_args()


def build_app_config(args: Namespace) -> AppConfig:
    """build runtime config"""
    return AppConfig(
        save_to=args.save_to,
        api_key=resolve_api_key(
            args.api_key,
            args.api_key_file,
            MATCHSTAT_API_KEY_ENV,
            "matchstat rapidapi",
        ),
        start_year=args.start_year,
        include_futures=args.include_futures,
        tournament_info=not args.no_tournament_info,
        output_file=args.output_file,
    )


def main() -> None:
    """download matchstat tennis data"""
    config: AppConfig = build_app_config(parse_arguments())
    frames: list[pd.DataFrame] = []

    hist_by_div = fetch_historical(config)
    upcoming_by_div = fetch_upcoming(config)

    tournament_cache: TournamentCache = (
        build_tournament_cache(config, upcoming_by_div)
        if config.tournament_info
        else _load_tournament_cache(config.tournament_cache_path)
    )

    for div, matches in hist_by_div.items():
        frame = build_hist_frame(matches, div)
        if not frame.empty:
            update_status(f"historical {div}: {len(frame)} matches")
            frames.append(frame)

    for div, fixtures in upcoming_by_div.items():
        frame = build_frame(
            fixtures, div, tournament_cache, is_upcoming=True
        )
        if not frame.empty:
            update_status(f"upcoming {div}: {len(frame)} matches")
            frames.append(frame)

    if not frames:
        raise RuntimeError(
            "no data fetched; check api key, date range, and network"
        )

    df: pd.DataFrame = pd.concat(frames, ignore_index=True)
    for col in ("SERIES_WINNER", "SERIES_LOSER"):
        df[col] = df[col].replace("", "UNKNOWN").fillna("UNKNOWN")
    df = uppercase_series_columns(df)
    df = smart_type(df)
    df = add_series_id_columns(df)
    df = finalize_grammar_frame(df)

    save_file: str = str(config.output_path)
    update_status(
        f"saving {len(df)} matches to {save_file}, shape: {df.shape}"
    )
    info_buffer: io.StringIO = io.StringIO()
    df.info(buf=info_buffer, verbose=True)
    save_parquet(df, save_file)
    update_status(info_buffer.getvalue())


if __name__ == "__main__":
    main()
