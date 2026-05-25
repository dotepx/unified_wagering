from __future__ import annotations

import io
import re
import threading
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date, timedelta
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
    parallelize_threads,
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
RANKING_PAGE_SIZE: Final[int] = 500
MATCH_PAGE_SIZE: Final[int] = 500
FIXTURE_PAGE_SIZE: Final[int] = 500
REQUEST_SPACING: Final[float] = 0.22  # under 5 requests/second cap

# included fields when fetching upcoming fixtures
FIXTURE_INCLUDE: Final[str] = (
    "round,tournament,tournament.court"
    ",tournament.rank,tournament.country,h2h"
)

# included fields when fetching historical player match archives
PAST_MATCH_INCLUDE: Final[str] = (
    "round,tournament,tournament.court"
    ",tournament.rank,tournament.country"
)

TOURNAMENT_CACHE_FILE: Final[str] = "tournament_info_cache.json"
PLAYER_MATCH_CACHE_DIR: Final[str] = "matchstat_player_past_matches"
RUN_STATE_FILE: Final[str] = "matchstat_loader_state.json"

# tiers excluded when include_futures is False
SKIP_TIERS: Final[frozenset[str]] = frozenset({"FUTURE", "FUTURES"})

INDOOR_MARKERS: Final[frozenset[str]] = frozenset(
    {"INDOOR HARD", "INDOOR CLAY", "CARPET"}
)
SURFACE_MAP: Final[dict[str, str]] = {
    "INDOOR HARD": "Hard",
    "INDOOR CLAY": "Clay",
}

SET_RE: re.Pattern = re.compile(r"(\d+)-(\d+)(?:\(\d+\))?")
RETIRED_RE: re.Pattern = re.compile(r"\bRET(?:IRED)?\b", re.IGNORECASE)
WALKOVER_RE: re.Pattern = re.compile(
    r"\bW[/ ]?O\b|\bWALKOVER\b", re.IGNORECASE
)
NONDIGIT_RE: re.Pattern = re.compile(r"[^\d.]")

# fallback round-name mapping for when include=round is unavailable
ROUND_NAMES: Final[dict[int, str]] = {
    0: "Qualifying",
    1: "Final",
    2: "Semi-Final",
    3: "Quarter-Final",
    4: "R16",
    5: "R32",
    6: "R64",
    7: "R128",
    16: "Round Robin",
}

# module-level rate-limit clock shared across all calls in this process
next_call_at: list[float] = [0.0]
rate_limit_lock: threading.Lock = threading.Lock()

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
    workers: int = 8
    ranking_limit: int = 500
    upcoming_days: int = 5
    refresh_player_cache: bool = False
    full_load: bool = False
    timeout: tuple[float, float] = (5.0, 30.0)
    output_file: str = "tennis_all_matches.pqt"

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError(
                "missing matchstat api key; use --api_key, "
                f"--api_key_file, or set {MATCHSTAT_API_KEY_ENV} env var"
            )
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.ranking_limit < 1:
            raise ValueError("ranking_limit must be at least 1")
        if self.upcoming_days < 1:
            raise ValueError("upcoming_days must be at least 1")
        BaseAppConfig.__post_init__(self)
        self.player_match_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_path(self) -> Path:
        """return the output parquet path"""
        return self.output_dir / self.output_file

    @property
    def tournament_cache_path(self) -> Path:
        """return the tournament metadata cache path"""
        return self.output_dir / TOURNAMENT_CACHE_FILE

    @property
    def run_state_path(self) -> Path:
        """return the incremental refresh state path"""
        return self.output_dir / RUN_STATE_FILE

    @property
    def player_match_cache_dir(self) -> Path:
        """return the per-player historical match cache directory"""
        return (
            self.output_dir
            / PLAYER_MATCH_CACHE_DIR
            / f"start_{self.start_year}"
        )


@dataclass(frozen=True, slots=True)
class PlayerMatchTask:
    """hold one player-history fetch task"""

    config: AppConfig
    division: str
    player_id: int
    refresh: bool


@dataclass(frozen=True, slots=True)
class PlayerMatchResult:
    """hold one player-history fetch result"""

    player_id: int
    matches: list[dict[str, Any]]
    from_cache: bool
    success: bool


# ---------------------------------------------------------------------------
# api layer
# ---------------------------------------------------------------------------


def api_get(
    config: AppConfig, path: str, params: dict[str, Any]
) -> Any:
    """one rate-limited matchstat api call"""
    with rate_limit_lock:
        wait: float = next_call_at[0] - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        next_call_at[0] = time.monotonic() + REQUEST_SPACING

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


def has_next_page(response: Any, records: list[Any]) -> bool:
    """detect whether the api signals a next page"""
    if isinstance(response, dict) and "hasNextPage" in response:
        return bool(response["hasNextPage"])
    if isinstance(response, list) and response:
        last_response: Any = response[-1]
        if isinstance(last_response, dict) and "hasNextPage" in last_response:
            return bool(last_response["hasNextPage"])
    if isinstance(response, dict):
        for val in response.values():
            if isinstance(val, list) and val:
                last_response: Any = val[-1]
                if (
                    isinstance(last_response, dict)
                    and "hasNextPage" in last_response
                ):
                    return bool(last_response["hasNextPage"])
    last: Any = records[-1] if records else None
    if isinstance(last, dict) and "hasNextPage" in last:
        return bool(last["hasNextPage"])
    return bool(records)


def is_page_marker(record: dict[str, Any]) -> bool:
    """return whether a dict is a pagination marker, not a data row"""
    return "hasNextPage" in record and "id" not in record


def extract_records(
    response: Any, keys: tuple[str, ...] = ("data",)
) -> list[dict[str, Any]]:
    """pull a record list from common matchstat response shapes"""
    if isinstance(response, list):
        return [
            record
            for record in response
            if isinstance(record, dict) and not is_page_marker(record)
        ]
    if isinstance(response, dict):
        for key in keys:
            if isinstance(response.get(key), list):
                return [
                    record
                    for record in response[key]
                    if isinstance(record, dict)
                    and not is_page_marker(record)
                ]
    return []


def extract_fixtures(response: Any) -> list[dict[str, Any]]:
    """pull the fixture list from whatever shape the response takes"""
    return extract_records(response, ("data", "fixtures", "results"))


def fetch_fixture_pages(
    config: AppConfig,
    division: str,
    path_suffix: str,
) -> list[dict[str, Any]]:
    """page through all upcoming fixtures for one division endpoint"""
    all_fixtures: list[dict[str, Any]] = []
    page_no: int = 1
    while True:
        response: Any = api_get(
            config,
            f"{division}/fixtures/{path_suffix}",
            {
                "include": FIXTURE_INCLUDE,
                "filter": "PlayerGroup:singles",
                "pageSize": FIXTURE_PAGE_SIZE,
                "pageNo": page_no,
            },
        )
        if response is None:
            update_status(
                f"null response: {division} {path_suffix} page {page_no}"
            )
            break
        if page_no == 1 and not extract_fixtures(response):
            update_status(
                f"unexpected response shape: {str(response)[:300]}"
            )
        fixtures: list[dict[str, Any]] = extract_fixtures(response)
        all_fixtures.extend(fixtures)
        update_status(
            f"{division} {path_suffix} p{page_no}: "
            f"{len(fixtures)} fixtures (total: {len(all_fixtures)})"
        )
        if not has_next_page(response, fixtures):
            break
        page_no += 1
    return all_fixtures


# ---------------------------------------------------------------------------
# Tournament info cache (prize money / draw size for upcoming)
# ---------------------------------------------------------------------------


def load_tournament_cache(cache_path: Path) -> TournamentCache:
    """load persisted cache from disk; returns empty dict if missing"""
    if not cache_path.exists():
        return {}
    try:
        return load_json(str(cache_path))
    except Exception as ex:
        update_status(f"tournament cache unreadable ({ex}), resetting")
        return {}


def save_tournament_cache(
    cache: TournamentCache, cache_path: Path
) -> None:
    """save the tournament metadata cache"""
    write_json(str(cache_path), cache)


def load_run_state(config: AppConfig) -> dict[str, Any]:
    """load the incremental refresh state"""
    if not config.run_state_path.exists():
        return {}
    try:
        return load_json(str(config.run_state_path))
    except Exception as ex:
        update_status(f"run state unreadable ({ex}), starting fresh")
        return {}


def fixture_date(raw_value: Any) -> date | None:
    """return the date portion of a fixture timestamp"""
    raw_text: str = str(raw_value or "").strip()
    if len(raw_text) < 10:
        return None
    try:
        return date.fromisoformat(raw_text[:10])
    except ValueError:
        return None


def run_state_last_success(raw_state: dict[str, Any]) -> date | None:
    """return the last successful run date from state"""
    raw_date: Any = raw_state.get("last_success_date")
    return fixture_date(raw_date)


def upcoming_fixture_state(
    upcoming_by_division: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """return compact fixture rows for the next incremental refresh"""
    return [
        {
            "division": division,
            "date": str(fixture.get("date") or "")[:10],
            "fixture_id": fixture.get("id", ""),
            "player1_id": fixture.get(
                "player1Id", (fixture.get("player1") or {}).get("id", "")
            ),
            "player2_id": fixture.get(
                "player2Id", (fixture.get("player2") or {}).get("id", "")
            ),
        }
        for division, fixtures in upcoming_by_division.items()
        for fixture in fixtures
        if fixture
    ]


def save_run_state(
    config: AppConfig,
    upcoming_by_division: dict[str, list[dict[str, Any]]],
) -> None:
    """save incremental refresh state after a successful run"""
    write_json(
        str(config.run_state_path),
        {
            "last_success_date": date.today().isoformat(),
            "start_year": config.start_year,
            "ranking_limit": config.ranking_limit,
            "upcoming_days": config.upcoming_days,
            "upcoming_fixtures": upcoming_fixture_state(upcoming_by_division),
        },
    )


def state_player_ids_since_last_run(
    raw_state: dict[str, Any],
) -> dict[str, set[int]]:
    """return players from prior fixtures whose dates have passed"""
    last_success_date: date | None = run_state_last_success(raw_state)
    if last_success_date is None:
        return {division: set() for division in DIVISIONS}

    through_date: date = date.today() - timedelta(days=1)
    refresh_ids: dict[str, set[int]] = {division: set() for division in DIVISIONS}
    fixtures: Any = raw_state.get("upcoming_fixtures", [])
    if not isinstance(fixtures, list) or through_date < last_success_date:
        return refresh_ids

    for fixture in fixtures:
        if not isinstance(fixture, dict):
            continue
        division: str = str(fixture.get("division") or "").lower()
        played_date: date | None = fixture_date(fixture.get("date"))
        if division not in refresh_ids or played_date is None:
            continue
        if not last_success_date <= played_date <= through_date:
            continue
        for key in ("player1_id", "player2_id"):
            player_id: Any = fixture.get(key)
            if player_id:
                refresh_ids[division].add(int(player_id))
    return refresh_ids


def should_full_refresh(config: AppConfig, raw_state: dict[str, Any]) -> bool:
    """return whether all ranked player histories should refresh"""
    return (
        config.refresh_player_cache
        or config.full_load
        or run_state_last_success(raw_state) is None
    )


def fetch_tournament_info(
    config: AppConfig,
    division: str,
    season_id: int,
    cache: TournamentCache,
    cache_path: Path,
) -> TournamentMeta:
    """fetch and cache one tournament season's metadata"""
    key: str = f"{division}/{season_id}"
    if key in cache:
        return cache[key]

    response: Any = api_get(
        config, f"{division}/tournament/info/{season_id}", {}
    )
    data: dict = response if isinstance(response, dict) else {}

    raw_prize: Any = data.get("prize", data.get("singlesPrize", ""))
    try:
        prize: float = (
            float(NONDIGIT_RE.sub("", str(raw_prize)))
            if raw_prize
            else np.nan
        )
    except ValueError:
        prize = np.nan

    raw_draw: Any = data.get(
        "draw_size", data.get("drawSize", data.get("draw", ""))
    )
    try:
        draw_size: float = float(str(raw_draw)) if raw_draw else np.nan
    except ValueError:
        draw_size = np.nan

    # api has a typo: "coutry" not "country"
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
    save_tournament_cache(cache, cache_path)
    return meta


def build_tournament_cache(
    config: AppConfig,
    upcoming_by_division: dict[str, list[dict[str, Any]]],
) -> TournamentCache:
    """fetch info for every unique (division, tournament_id) in upcoming"""
    cache: TournamentCache = load_tournament_cache(
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
        fetch_tournament_info(
            config, div, season_id, cache, config.tournament_cache_path
        )

    return cache


# ---------------------------------------------------------------------------
# Shared row-building helpers
# ---------------------------------------------------------------------------


def court_surface(court_name: str) -> tuple[str, str]:
    """return court and surface from matchstat court name"""
    upper: str = (court_name or "").strip().upper()
    is_indoor: bool = upper in INDOOR_MARKERS or "INDOOR" in upper
    court: str = "Indoor" if is_indoor else "Outdoor"
    surface: str = SURFACE_MAP.get(upper, court_name.strip()) or ""
    return court, surface


def upper_text(raw_value: Any) -> str:
    """return an uppercase stripped string"""
    return str(raw_value or "").strip().upper()


def best_of(rank_name: str, division: str) -> int:
    """only atp grand slams are best of five"""
    return (
        5
        if "GRAND SLAM" in rank_name.upper() and division == "atp"
        else 3
    )


def parse_seed(raw: Any) -> float:
    """numeric seeds only; wc, q, ll, and pr become nan"""
    if raw is None or str(raw).strip() == "":
        return np.nan
    try:
        return float(str(raw).strip())
    except ValueError:
        return np.nan


def parse_numeric(raw: Any) -> float:
    """parse provider numeric strings, returning nan when absent"""
    if raw is None or str(raw).strip() == "":
        return np.nan
    try:
        return float(NONDIGIT_RE.sub("", str(raw)))
    except ValueError:
        return np.nan


def parse_best_of(raw: Any, rank_name: str, division: str) -> int:
    """parse best-of, falling back to the tournament level rule"""
    if raw is None or str(raw).strip() == "":
        return best_of(rank_name, division)
    try:
        return int(float(str(raw).strip()))
    except ValueError:
        return best_of(rank_name, division)


def parse_result(
    result: str,
    winner_is_player1: bool = True,
) -> dict[str, Any]:
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

    if WALKOVER_RE.search(result):
        row["CATEGORY_COMMENT"] = "WALKOVER"
    elif RETIRED_RE.search(result):
        row["CATEGORY_COMMENT"] = "RETIRED"
    else:
        row["CATEGORY_COMMENT"] = "COMPLETED"

    sets: list[tuple[str, str]] = SET_RE.findall(result)
    w_wins: int = 0
    l_wins: int = 0
    for i, (p1s, p2s) in enumerate(sets[:5], start=1):
        winner_score, loser_score = (
            (int(p1s), int(p2s))
            if winner_is_player1
            else (int(p2s), int(p1s))
        )
        row[f"FEATURE_WINNER_SET_{i}"] = winner_score
        row[f"FEATURE_LOSER_SET_{i}"] = loser_score
        if winner_score > loser_score:
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
    """map one upcoming api fixture to a grammar row"""
    tournament: dict = fixture.get("tournament") or {}
    rank: dict = tournament.get("rank") or {}
    court: dict = tournament.get("court") or {}
    p1: dict = fixture.get("player1") or {}
    p2: dict = fixture.get("player2") or {}
    rnd: dict = fixture.get("round") or {}
    h2h: dict = fixture.get("h2h") or {}
    meta: TournamentMeta = tournament_meta or {}

    raw_surface: str = court.get("name", "") or meta.get("surface", "")
    court_val, surface_val = court_surface(raw_surface)
    rank_name: str = rank.get("name", "")

    country: dict = tournament.get("country") or {}
    location: str = (
        meta.get("site")
        or str(country.get("name", "") or "").strip()
        or tournament.get("countryAcr", "")
    )

    # H2H is career-total and would leak future results into historical
    # training rows; only populate it for upcoming matches.
    h2h_p1: float = (
        float(h2h["player1AllWins"]) if is_upcoming and h2h else np.nan
    )
    h2h_p2: float = (
        float(h2h["player2AllWins"]) if is_upcoming and h2h else np.nan
    )

    row: dict[str, Any] = {
        "LABEL_DATE": (fixture.get("date") or "")[:10],
        "LABEL_SOURCE_FIXTURE_ID": fixture.get("id", ""),
        "LABEL_SOURCE_TOURNAMENT_ID": fixture.get(
            "tournamentId", tournament.get("id", "")
        ),
        "LABEL_SOURCE_ROUND_ID": fixture.get("roundId", ""),
        "LABEL_SOURCE_PLAYER1_ID": fixture.get("player1Id", p1.get("id", "")),
        "LABEL_SOURCE_PLAYER2_ID": fixture.get("player2Id", p2.get("id", "")),
        "LABEL_COMMENCE_TIME": (fixture.get("date") or ""),
        "SERIES_WINNER": upper_text(p1.get("name")),
        "SERIES_LOSER": upper_text(p2.get("name")),
        "FEATURE_WINNER_RANK": np.nan,
        "FEATURE_LOSER_RANK": np.nan,
        "FEATURE_WINNER_POINTS": np.nan,
        "FEATURE_LOSER_POINTS": np.nan,
        "FEATURE_P1_SEED": parse_seed(fixture.get("seed1")),
        "FEATURE_P2_SEED": parse_seed(fixture.get("seed2")),
        "FEATURE_P1_H2H_WINS": h2h_p1,
        "FEATURE_P2_H2H_WINS": h2h_p2,
        "FEATURE_DRAW_SIZE": meta.get("draw_size", np.nan),
        "FEATURE_PRIZE_MONEY": meta.get("prize", np.nan),
        "CATEGORY_LOCATION": upper_text(location),
        "CATEGORY_TOURNAMENT": upper_text(tournament.get("name")),
        "CATEGORY_COURT": upper_text(court_val),
        "CATEGORY_SURFACE": upper_text(surface_val),
        "CATEGORY_ROUND": upper_text(rnd.get("name")),
        "CATEGORY_SETS": best_of(rank_name, division),
        "CATEGORY_SERIES": upper_text(rank_name),
        "CATEGORY_MATCH_DIVISION": division.upper(),
        "CATEGORY_P1_COUNTRY": upper_text(p1.get("countryAcr")),
        "CATEGORY_P2_COUNTRY": upper_text(p2.get("countryAcr")),
    }
    row.update(parse_result(fixture.get("result", "")))
    return row


def build_frame(
    fixtures: list[dict[str, Any]],
    division: str,
    tournament_cache: TournamentCache,
    is_upcoming: bool,
) -> pd.DataFrame:
    """convert upcoming fixtures to a dataframe"""
    rows: list[dict[str, Any]] = [
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
# Historical player past-matches -> row
# ---------------------------------------------------------------------------


def year_filter(config: AppConfig) -> str:
    """return the api filter for the configured historical year range"""
    this_year: int = date.today().year
    years: str = ",".join(
        str(year) for year in range(config.start_year, this_year + 1)
    )
    return f"GameYear:{years}" if years else "GameYear:"


def match_key(division: str, match: dict[str, Any]) -> str:
    """return a stable key for deduping player history rows"""
    source_id: Any = match.get("id")
    if source_id:
        return f"{division}:{source_id}"
    return (
        f"{division}:{match.get('date')}:{match.get('player1Id')}:"
        f"{match.get('player2Id')}:{match.get('tournamentId')}:"
        f"{match.get('roundId')}:{match.get('result')}"
    )


def included_history_match(
    match: dict[str, Any],
    include_futures: bool,
) -> bool:
    """return whether a historical match passes loader-level filters"""
    if include_futures:
        return True
    tournament: dict = match.get("tournament") or {}
    rank: dict = tournament.get("rank") or {}
    tier: str = str(rank.get("name") or tournament.get("tier") or "").upper()
    return not any(skip in tier for skip in SKIP_TIERS)


def ranking_player(entry: dict[str, Any]) -> dict[str, Any] | None:
    """return a normalized player dict from one ranking entry"""
    player: Any = entry.get("player")
    if isinstance(player, dict) and player.get("id"):
        return {
            "id": int(player["id"]),
            "name": player.get("name", ""),
            "countryAcr": player.get("countryAcr", ""),
            "position": entry.get("position", ""),
        }
    if entry.get("id"):
        return {
            "id": int(entry["id"]),
            "name": entry.get("name", ""),
            "countryAcr": entry.get("countryAcr", ""),
            "position": entry.get("position", ""),
        }
    return None


def fetch_ranked_players(
    config: AppConfig,
    division: str,
) -> list[dict[str, Any]]:
    """fetch active singles players from the live ranking endpoint"""
    players: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    page_no: int = 1
    while len(players) < config.ranking_limit:
        page_size: int = min(
            RANKING_PAGE_SIZE, config.ranking_limit - len(players)
        )
        response: Any = api_get(
            config,
            f"{division}/ranking/singles",
            {
                "pageSize": page_size,
                "pageNo": page_no,
            },
        )
        page_rankings: list[dict[str, Any]] = extract_records(response)
        if response is None:
            update_status(f"null ranking response: {division} page {page_no}")
            break
        if page_no == 1 and not page_rankings:
            update_status(f"unexpected ranking response: {str(response)[:300]}")
        page_players: list[dict[str, Any]] = [
            player
            for entry in page_rankings
            if (player := ranking_player(entry)) is not None
            and player["id"] not in seen_ids
        ]
        seen_ids.update(player["id"] for player in page_players)
        players.extend(page_players)
        update_status(
            f"{division} ranking p{page_no}: "
            f"{len(page_players)} players (total: {len(players)})"
        )
        if not has_next_page(response, page_rankings):
            break
        page_no += 1
    return players


def fetch_player_past_matches(
    config: AppConfig,
    division: str,
    player_id: int,
) -> list[dict[str, Any]] | None:
    """page through one player's completed match archive"""
    matches: list[dict[str, Any]] = []
    page_no: int = 1
    while True:
        response: Any = api_get(
            config,
            f"{division}/player/past-matches/{player_id}",
            {
                "include": PAST_MATCH_INCLUDE,
                "filter": year_filter(config),
                "pageSize": MATCH_PAGE_SIZE,
                "pageNo": page_no,
            },
        )
        page_matches: list[dict[str, Any]] = extract_records(response)
        if response is None:
            update_status(
                f"null past-matches response: {division} player {player_id} "
                f"page {page_no}"
            )
            return None
        matches.extend(page_matches)
        if not has_next_page(response, page_matches):
            break
        page_no += 1
    return matches


def player_cache_path(
    config: AppConfig,
    division: str,
    player_id: int,
) -> Path:
    """return the per-player match cache path"""
    division_dir: Path = config.player_match_cache_dir / division
    division_dir.mkdir(parents=True, exist_ok=True)
    return division_dir / f"{player_id}.json"


def is_daily_cache_fresh(path: Path) -> bool:
    """return whether a cache file was updated today"""
    return path.exists() and date.fromtimestamp(path.stat().st_mtime) == date.today()


def load_player_match_cache(path: Path) -> list[dict[str, Any]] | None:
    """load one per-player match cache"""
    if not path.exists():
        return None
    try:
        payload: dict[str, Any] = load_json(str(path))
    except Exception as ex:
        update_status(f"player cache unreadable ({path.name}): {ex}")
        return None
    matches: Any = payload.get("matches") if isinstance(payload, dict) else None
    if isinstance(matches, list):
        return [match for match in matches if isinstance(match, dict)]
    update_status(f"player cache has unexpected shape: {path}")
    return None


def save_player_match_cache(
    path: Path,
    division: str,
    player_id: int,
    config: AppConfig,
    matches: list[dict[str, Any]],
) -> None:
    """save one player's raw past-match records"""
    write_json(
        str(path),
        {
            "division": division,
            "player_id": player_id,
            "start_year": config.start_year,
            "fetched_date": date.today().isoformat(),
            "matches": matches,
        },
    )


def load_or_fetch_player_past_matches(
    task: PlayerMatchTask,
) -> PlayerMatchResult:
    """load today's player cache or refresh it from the api"""
    config: AppConfig = task.config
    division: str = task.division
    player_id: int = task.player_id
    cache_path: Path = player_cache_path(config, division, player_id)
    cached_matches: list[dict[str, Any]] | None = load_player_match_cache(
        cache_path
    )
    if not task.refresh and cached_matches is not None:
        return PlayerMatchResult(player_id, cached_matches, True, True)

    fetched_matches: list[dict[str, Any]] | None = fetch_player_past_matches(
        config, division, player_id
    )
    if fetched_matches is None:
        if cached_matches is not None:
            update_status(
                f"using stale cache for {division} player {player_id}"
            )
            return PlayerMatchResult(player_id, cached_matches, True, True)
        return PlayerMatchResult(player_id, [], False, False)

    save_player_match_cache(
        cache_path, division, player_id, config, fetched_matches
    )
    return PlayerMatchResult(player_id, fetched_matches, False, True)


def safe_load_or_fetch_player_past_matches(
    task: PlayerMatchTask,
) -> PlayerMatchResult:
    """load or fetch player matches without raising into the worker pool"""
    try:
        return load_or_fetch_player_past_matches(task)
    except Exception as ex:
        update_status(f"{task.division} player {task.player_id} failed: {ex}")
        return PlayerMatchResult(task.player_id, [], False, False)


def result_to_row(
    match: dict[str, Any],
    division: str,
) -> dict[str, Any] | None:
    """convert one player past-match record to a grammar row"""
    p1: dict = match.get("player1") or {}
    p2: dict = match.get("player2") or {}
    p1_id: Any = match.get("player1Id") or p1.get("id")
    p2_id: Any = match.get("player2Id") or p2.get("id")
    winner_id: Any = match.get("match_winner")

    if winner_id == p2_id:
        winner, loser = p2, p1
        winner_source_id, loser_source_id = p2_id, p1_id
    elif winner_id in (None, p1_id):
        winner, loser = p1, p2
        winner_source_id, loser_source_id = p1_id, p2_id
    else:
        return None

    rnd: dict = match.get("round") or {}
    round_id: int = int(match.get("roundId") or 0)
    round_name: str = (
        rnd.get("name") or ROUND_NAMES.get(round_id, f"Round_{round_id}")
    )

    tournament: dict = match.get("tournament") or {}
    court_obj: dict = tournament.get("court") or {}
    country_obj: dict = (
        tournament.get("country") or tournament.get("coutry") or {}
    )
    rank_obj: dict = tournament.get("rank") or {}
    rank_name: str = str(rank_obj.get("name") or tournament.get("tier") or "")

    raw_surface: str = court_obj.get("name", "")
    court_val, surface_val = court_surface(raw_surface)
    location: str = (
        country_obj.get("name")
        or tournament.get("countryAcr")
        or tournament.get("site")
        or ""
    )

    row: dict[str, Any] = {
        "LABEL_DATE": (match.get("date") or "")[:10],
        "LABEL_SOURCE_FIXTURE_ID": match.get("id", ""),
        "LABEL_SOURCE_TOURNAMENT_ID": match.get(
            "tournamentId", tournament.get("id", "")
        ),
        "LABEL_SOURCE_ROUND_ID": round_id,
        "LABEL_SOURCE_PLAYER1_ID": p1_id or "",
        "LABEL_SOURCE_PLAYER2_ID": p2_id or "",
        "LABEL_SOURCE_WINNER_PLAYER_ID": winner_source_id or "",
        "LABEL_SOURCE_LOSER_PLAYER_ID": loser_source_id or "",
        "LABEL_COMMENCE_TIME": (match.get("date") or ""),
        "SERIES_WINNER": upper_text(winner.get("name")),
        "SERIES_LOSER": upper_text(loser.get("name")),
        "FEATURE_WINNER_RANK": np.nan,
        "FEATURE_LOSER_RANK": np.nan,
        "FEATURE_WINNER_POINTS": np.nan,
        "FEATURE_LOSER_POINTS": np.nan,
        "FEATURE_P1_SEED": np.nan,
        "FEATURE_P2_SEED": np.nan,
        "FEATURE_P1_H2H_WINS": np.nan,
        "FEATURE_P2_H2H_WINS": np.nan,
        "FEATURE_DRAW_SIZE": parse_numeric(tournament.get("draw_size")),
        "FEATURE_PRIZE_MONEY": parse_numeric(tournament.get("prize")),
        "CATEGORY_LOCATION": upper_text(location),
        "CATEGORY_TOURNAMENT": upper_text(tournament.get("name")),
        "CATEGORY_COURT": upper_text(court_val),
        "CATEGORY_SURFACE": upper_text(surface_val),
        "CATEGORY_ROUND": upper_text(round_name),
        "CATEGORY_SETS": parse_best_of(
            match.get("best_of"), rank_name, division
        ),
        "CATEGORY_SERIES": upper_text(rank_name),
        "CATEGORY_MATCH_DIVISION": division.upper(),
        "CATEGORY_P1_COUNTRY": upper_text(winner.get("countryAcr")),
        "CATEGORY_P2_COUNTRY": upper_text(loser.get("countryAcr")),
    }
    row.update(
        parse_result(
            match.get("result", ""),
            winner_is_player1=winner_source_id == p1_id,
        )
    )
    return row


def build_hist_frame(
    matches: list[dict[str, Any]],
    division: str,
) -> pd.DataFrame:
    """convert player past-match records to a dataframe"""
    rows: list[dict[str, Any]] = [
        row
        for match in matches
        if (row := result_to_row(match, division)) is not None
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_historical(
    config: AppConfig,
    raw_state: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """fetch completed matches via every player's past-matches archive"""
    by_division: dict[str, list[dict[str, Any]]] = {d: [] for d in DIVISIONS}
    state_refresh_ids: dict[str, set[int]] = state_player_ids_since_last_run(
        raw_state
    )
    full_refresh: bool = should_full_refresh(config, raw_state)

    for division in DIVISIONS:
        players: list[dict[str, Any]] = fetch_ranked_players(config, division)
        player_ids: list[int] = [
            player["id"]
            for player in players
            if player.get("id")
        ]
        refresh_ids: set[int] = (
            set(player_ids)
            if full_refresh
            else state_refresh_ids[division]
        )
        update_status(
            f"loading {division} histories for {len(player_ids)} ranked players; "
            f"refreshing {len(refresh_ids)} from match calendar "
            f"with {config.workers} workers"
        )
        match_by_key: dict[str, dict[str, Any]] = {}
        cache_hits: int = 0
        failures: int = 0
        tasks: list[PlayerMatchTask] = [
            PlayerMatchTask(config, division, player_id, player_id in refresh_ids)
            for player_id in player_ids
        ]
        results: list[PlayerMatchResult] = parallelize_threads(
            safe_load_or_fetch_player_past_matches,
            tasks,
            config.workers,
            desc=f"{division} player histories",
        )
        for idx, result in enumerate(results, start=1):
            cache_hits += int(result.from_cache)
            failures += int(not result.success)
            kept_matches: list[dict[str, Any]] = [
                match
                for match in result.matches
                if included_history_match(match, config.include_futures)
            ]
            for match in kept_matches:
                match_by_key.setdefault(match_key(division, match), match)
            if idx == 1 or idx % 25 == 0 or idx == len(results):
                update_status(
                    f"{division} players {idx}/{len(results)}: "
                    f"{len(match_by_key)} unique matches, "
                    f"{cache_hits} cache hits, {failures} misses"
                )
        by_division[division] = list(match_by_key.values())

    return by_division


def fetch_upcoming(
    config: AppConfig,
) -> dict[str, list[dict[str, Any]]]:
    """fetch unplayed matches for the configured upcoming date window"""
    start_date: date = date.today()
    end_date: date = start_date + timedelta(days=config.upcoming_days - 1)
    update_status(
        f"fetching upcoming {start_date.isoformat()} to "
        f"{end_date.isoformat()}"
    )
    by_division: dict[str, list[dict[str, Any]]] = {d: [] for d in DIVISIONS}
    for division in DIVISIONS:
        path_suffix: str = (
            start_date.isoformat()
            if config.upcoming_days == 1
            else f"{start_date.isoformat()}/{end_date.isoformat()}"
        )
        raw: list[dict[str, Any]] = fetch_fixture_pages(
            config, division, path_suffix
        )
        by_division[division].extend(
            fixture for fixture in raw if fixture.get("complete") is None
        )
        update_status(
            f"upcoming {division}: {len(by_division[division])} fixtures "
            f"over {config.upcoming_days} days"
        )
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
        "--workers",
        type=int,
        default=8,
        help="parallel player-history workers (default 8)",
    )
    parser.add_argument(
        "--ranking_limit",
        type=int,
        default=500,
        help="top ranked singles players per division to fetch (default 500)",
    )
    parser.add_argument(
        "--upcoming_days",
        type=int,
        default=5,
        help="number of calendar days of upcoming fixtures to fetch (default 5)",
    )
    parser.add_argument(
        "--refresh_player_cache",
        action="store_true",
        help="force refresh of per-player past-match cache files",
    )
    parser.add_argument(
        "--full_load",
        "--womp",
        action="store_true",
        help="refresh every ranked player history on demand",
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
        workers=args.workers,
        ranking_limit=args.ranking_limit,
        upcoming_days=args.upcoming_days,
        refresh_player_cache=args.refresh_player_cache,
        full_load=args.full_load,
        output_file=args.output_file,
    )


def main() -> None:
    """download matchstat tennis data"""
    config: AppConfig = build_app_config(parse_arguments())
    frames: list[pd.DataFrame] = []
    raw_state: dict[str, Any] = load_run_state(config)

    hist_by_div: dict[str, list[dict[str, Any]]] = fetch_historical(
        config, raw_state
    )
    upcoming_by_div: dict[str, list[dict[str, Any]]] = fetch_upcoming(config)

    tournament_cache: TournamentCache = (
        build_tournament_cache(config, upcoming_by_div)
        if config.tournament_info
        else load_tournament_cache(config.tournament_cache_path)
    )

    for div, matches in hist_by_div.items():
        frame: pd.DataFrame = build_hist_frame(matches, div)
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
    save_run_state(config, upcoming_by_div)
    update_status(info_buffer.getvalue())


if __name__ == "__main__":
    main()
