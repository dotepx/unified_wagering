# Matchstat Loader

`matchstat_loader.py` downloads ATP and WTA tennis match data from the
Matchstat RapidAPI service and saves it in the project grammar used by the
modeling pipeline.

This document is an explainer for the loader's expected behavior and operating
commands. It is not a changelog.

## What It Fetches

Historical matches are built from the current live singles rankings:

```text
GET /tennis/v2/atp/ranking/singles?pageSize=500
GET /tennis/v2/wta/ranking/singles?pageSize=500
```

The loader extracts each ranked player's provider ID from `player.id`, then
fetches that player's match history:

```text
GET /tennis/v2/{atp|wta}/player/past-matches/{playerId}?pageSize=500
```

Historical match requests include nested tournament metadata:

```text
include=round,tournament,tournament.court,tournament.rank,tournament.country
```

Upcoming fixtures are fetched from:

```text
GET /tennis/v2/{atp|wta}/fixtures/{startdate}/{enddate}?pageSize=500
```

The default upcoming fixture window is 5 calendar days: today plus the next 4
days.

```bash
--upcoming_days 5
```

## Reference Tables

Matchstat also exposes lookup/reference endpoints:

```text
GET /tennis/v2/court
GET /tennis/v2/ranking
GET /tennis/v2/round
GET /tennis/v2/countries
```

The loader relies on nested includes for these values instead of joining
separate lookup tables:

```text
tournament.court
tournament.rank
tournament.country
round
```

Provider reference IDs are preserved where they appear in the match or fixture
payload, for example `LABEL_SOURCE_ROUND_ID` and
`LABEL_SOURCE_TOURNAMENT_ID`.

## Pagination

The loader requests the largest practical page size used by the service:

```text
ranking singles: 500 rows per page
player past matches: 500 rows per page
upcoming fixtures: 500 rows per page
```

Pagination continues while the response reports `hasNextPage`.

## Player Universe

The default player universe is the top 500 ranked ATP singles players and the
top 500 ranked WTA singles players.

```bash
--ranking_limit 500
```

The loader does not increment or guess player IDs. It only uses IDs returned by
the ranking endpoint.

## Per-Player Cache

Each player history response is cached as raw JSON. Normal daily runs load all
ranked player histories from cache, then refresh only players from prior
stored upcoming fixtures whose scheduled dates have passed since the last
successful run.

Default cache layout:

```text
data/matchstat_player_past_matches/start_2007/atp/{player_id}.json
data/matchstat_player_past_matches/start_2007/wta/{player_id}.json
```

The `start_2007` directory changes with `--start_year`, which keeps short test
runs separate from full-history runs.

Use this to force refresh every ranked player history:

```bash
--full_load
```

`--womp` is an alias for `--full_load`. `--refresh_player_cache` is retained as
an equivalent explicit cache-refresh switch.

## Surface And Court

For historical player matches, surface comes from:

```text
match.tournament.court.name
```

For upcoming fixtures, surface comes from:

```text
fixture.tournament.court.name
```

If an upcoming fixture is missing embedded court data, the loader can fall back
to the tournament-info cache.

The raw provider court name is normalized by `_court_surface()` into:

```text
CATEGORY_SURFACE
CATEGORY_COURT
```

Examples:

```text
Clay        -> CATEGORY_SURFACE=CLAY, CATEGORY_COURT=OUTDOOR
Grass       -> CATEGORY_SURFACE=GRASS, CATEGORY_COURT=OUTDOOR
Hard        -> CATEGORY_SURFACE=HARD, CATEGORY_COURT=OUTDOOR
Indoor Hard -> CATEGORY_SURFACE=HARD, CATEGORY_COURT=INDOOR
Indoor Clay -> CATEGORY_SURFACE=CLAY, CATEGORY_COURT=INDOOR
Carpet      -> CATEGORY_SURFACE=CARPET, CATEGORY_COURT=INDOOR
```

If the provider returns another value such as `Concrete`, the loader preserves
that value as the surface after normal string cleanup.

## Output

The final combined parquet is saved to:

```text
{save_to}/tennis_all_matches.pqt
```

Override the filename with:

```bash
--output_file matchstat_all_matches.pqt
```

Before saving, the loader:

- uppercases string `SERIES_` columns,
- adds stable custom `SERIES_*_ID` columns,
- smart-casts data types,
- validates and orders columns by the project grammar.

## Typical WSL Command

```bash
cd /mnt/c/Users/ximpact/Documents/unified_wagering
source ~/.bashrc
python matchstat_loader.py --start_year 2007 --save_to data --ranking_limit 500 --upcoming_days 5 --workers 8
```

For a smaller test:

```bash
python matchstat_loader.py --start_year 2025 --save_to data --ranking_limit 50 --output_file matchstat_test.pqt
```

For an on-demand full reload:

```bash
python matchstat_loader.py --start_year 2007 --save_to data --ranking_limit 500 --upcoming_days 5 --workers 8 --womp
```

## Important Notes

- `MATCHSTAT_API_KEY` must be exported into the shell environment.
- `--workers` controls concurrent player-history fetches; the default is 8.
- API calls still pass through a shared rate limiter, currently set just under
  5 requests per second.
- The plan has a 10k request/month cap, so incremental daily refreshes are
  important.
- Historical player histories are deduped by provider match ID because each
  match can appear in both players' histories.
- Loader output is target-free; `TARGET_` columns belong in later feature or
  labeling stages.
