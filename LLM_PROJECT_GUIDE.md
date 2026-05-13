# Unified Wager LLM Project Guide

This file is a fast-start guide for an LLM working in this repository. The
source of truth for coding rules is `CONVENTIONS.md`; this guide explains the
project shape and the decisions an assistant should preserve while editing.

## Project Goal

Unified Wager is a dataset-agnostic wagering and forecasting tool. It should be
able to ingest heterogeneous domains such as equities, macro series, NCAA men's
basketball, global football, and tennis, normalize them into one stable grammar,
then build model-ready datasets.

The prediction direction is:

- regress a selected numeric target value;
- learn probabilities for bucketed target outcomes;
- keep the engine generic enough that the same modeling code can consume many
  domains once their data follows the grammar.

In this repo snapshot, loaders and dataset builders are the main implemented
layers. A standalone training engine is not yet the central artifact, so new
modeling work should consume the existing grammar rather than inventing a
domain-specific schema.

## Core Data Grammar

Every normalized dataframe at a pipeline boundary must use these prefixes:

- `LABEL_`: row context and human-readable metadata. Prefer `LABEL_DATE` for
  time.
- `SERIES_`: modeled entity identity, such as `SERIES_SYMBOL`, `SERIES_HOME`,
  `SERIES_AWAY`, `SERIES_TEAM`, or `SERIES_PLAYER`.
- `FEATURE_`: numeric model inputs available at prediction time.
- `TARGET_`: values defined by a later feature-building or labeling stage for
  prediction or target-distribution learning.
- `IS_`: binary flags.
- `CATEGORY_`: categorical variables.

Use `utilities.finalize_grammar_frame` before saving normalized outputs. It
orders grammar columns and fails if a column sits outside the grammar.

Important modeling distinction: a raw regression target should be a numeric
`TARGET_*` column that is not bucket metadata. Bucket metadata also starts with
`TARGET_`, so model code should distinguish:

- regression targets: examples like `TARGET_EXCESS_LOG_RETURN_5`,
  `TARGET_DELTA_SCORE`, `TARGET_TOTAL_SCORE`;
- bucket ids: `TARGET_TYPE_<suffix>`;
- bucket class labels: `TARGET_CLASS_<suffix>`;
- one-vs-rest bucket indicators: `TARGET_BUCKET_<suffix>_<nn>`.

## Layering Rules

Loaders are raw-only and grammar-normalized. They should fetch, clean, type,
validate, normalize names, add stable IDs when useful, and save parquet. They
should not define `TARGET_` columns or derivative historical features.

Dataset builders are where targets and derivative features belong. They may add
rolling, lagged, EWM, relative, forward-return, bucket, or other target-aware
formulations.

Modeling code should be dataset-agnostic. It should discover feature columns
from `FEATURE_`, optional categorical inputs from `CATEGORY_` and `IS_`, and
targets from explicitly selected numeric `TARGET_*` columns. Do not assume that
a domain is stocks, football, tennis, or basketball unless the caller selects a
domain-specific dataset.

## Stable IDs

String or object `SERIES_` columns should usually get deterministic unsigned
integer ID companions when they help joins, grouping, or modeling. The standard
form is `SERIES_<NAME>_ID`.

Use the helpers in `utilities.py`:

- `uppercase_series_columns`
- `add_series_id_columns`
- `make_stable_uint_ids`
- `make_id_column_name`

Provider-supplied IDs are source metadata, not custom series IDs. Preserve those
as `LABEL_SOURCE_*_ID` when needed.

## Current Module Map

Shared utilities:

- `CONVENTIONS.md`: required style, grammar, loader, and prediction rules.
- `utilities.py`: shared CLI helpers, API-key resolution, retries, parquet IO,
  grammar validation/finalization, stable ID generation, logging, and typing.
- `market_data.py`: shared market column constants.
- `feature_building.py`: reusable feature/target helpers, including calendar
  categories, guarded log ratios, EWM features, and positive quantile target
  buckets.
- `poc_model.py`: generic proof-of-concept model trainer for grammar-normalized
  datasets. It trains one regression head for a selected numeric `TARGET_*`
  column and, when matching bucket columns exist, one classifier head for target
  bucket probabilities.
- `market_predictor.py`: LightGBM-based bucket probability predictor for market
  datasets. Discovers all `TARGET_TYPE_*` columns, trains one multiclass model
  per target period using a time-safe date split, saves native `.lgb` boosters
  with JSON metadata sidecars, and optionally writes a predictions parquet with
  `PRED_BUCKET_*` columns.
- `market_sequence_predictor.py`: TensorFlow residual 1D CNN bucket predictor.
  Slides a `--seq_len` window over each symbol's sorted time series to build
  `(seq_len, n_features)` input sequences. Uses only `FEATURE_*` columns (no
  calendar features) so the network learns temporal patterns organically. Fits
  per-feature z-score normalization on training sequences only. Architecture:
  initial pointwise projection → stacked residual blocks (Conv1D + BN + ReLU ×2
  with skip connection) → GlobalAveragePooling1D → dense head → softmax. Saves
  `.keras` model files and `.json` normalization+metadata sidecars.

Market data:

- `av_loader.py`: Alpha Vantage daily adjusted price loader. Outputs raw,
  grammar-normalized per-symbol parquet files with market price features.
- `fred_loader.py`: FRED observations loader. Outputs raw, grammar-normalized
  per-series parquet files.
- `market_dataset_builder.py`: joins Alpha Vantage and FRED panels, adds
  calendar/external/price features, defines forward log-return targets, and adds
  quantile bucket targets.

Football:

- `footy_loader.py`: FootyStats API loader for global football match data.
  Normalizes raw match observations into grammar columns.
- `footy_dataset_builder.py`: pivots matches to team perspective, adds
  prediction-time-safe rolling features, defines score targets, and adds bucket
  targets.

Tennis:

- `tennis_loader.py`: tennis-data.co.uk historical loader with optional current
  match CSV append. Normalizes match rows and player series IDs.
- `matchstat_loader.py`: MatchStat/RapidAPI tennis loader for fixtures,
  tournament metadata, and results.
- `tennis_odds_scraper.py`: tennis odds scraper output support.
- `tennis_odds_output/`: local scraper output artifacts.

NCAA basketball:

- `ncaa_loader.py`: KenPom-backed NCAA men's basketball match loader. It
  normalizes teams, conferences, coaches, scores, completion flags, and neutral
  site flags into grammar columns.

## Target And Bucket Convention

Use `feature_building.add_positive_quantile_target_buckets` for the standard
bucket scheme unless a task explicitly needs a different distribution strategy.

For each numeric target `TARGET_<suffix>`, that helper creates:

- `TARGET_TYPE_<suffix>`: integer class id, with `0` for non-positive values
  and positive quantile buckets starting at `1`;
- `TARGET_CLASS_<suffix>`: class label such as `LTE_0` or `POS_Q01`;
- `TARGET_BUCKET_<suffix>_<nn>`: binary indicator for each bucket.

This supports a multi-head model pattern:

- a regression head predicts the numeric `TARGET_<suffix>`;
- a classification or calibration head predicts the probability distribution
  across `TARGET_BUCKET_<suffix>_*`.

Rows with unavailable future outcomes should keep numeric targets missing rather
than fabricating labels. Prediction-time rows may have features without targets.

## Adding A New Data Source

When adding a loader:

1. Create a script-style module with `parse_arguments`, `build_app_config`, and
   `main`.
2. Use `BaseAppConfig` and shared parser helpers from `utilities.py`.
3. Resolve API keys in this order: `--api_key`, `--api_key_file`, provider env
   var.
4. Fetch raw observations and handle expected external failures gracefully.
5. Rename columns into the grammar. Keep observed numeric values as `FEATURE_`
   unless a later builder defines them as targets.
6. Uppercase string `SERIES_` values and add stable `SERIES_*_ID` columns when
   useful.
7. Validate source contracts with Pandera or direct checks when appropriate.
8. Call `finalize_grammar_frame`, then save parquet.

When adding a dataset builder:

1. Load one or more normalized loader outputs.
2. Validate required input columns immediately.
3. Add prediction-time-safe features. Avoid look-ahead leakage: rolling values
   should shift before aggregating.
4. Define numeric `TARGET_` columns.
5. Add target buckets with `QuantileBucketSpec` when distribution learning is
   needed.
6. Validate uniqueness and required output columns.
7. Call `smart_type`, `finalize_grammar_frame`, and save parquet.

## Modeling Guidance

Future modeling code should treat the dataset contract as the interface:

- feature matrix: `FEATURE_*`, plus encoded `CATEGORY_*` and `IS_*` as selected;
- entity/time context: `SERIES_*` and `LABEL_*`;
- regression labels: caller-selected numeric `TARGET_*` columns;
- probability labels: matching `TARGET_BUCKET_*` columns grouped by target
  suffix.

Do not let model code hardcode a provider payload shape. Provider details should
end at the loader boundary. Do not let loaders decide which outcomes are the
betting target. Target definitions are part of feature building or labeling.

## Validation Philosophy

External systems can fail and should be handled with clear logs, retries, empty
frames, or failed-id files where appropriate. Internal contract violations
should fail fast with actionable errors.

Validate at pipeline boundaries, then trust the contract internally. Useful
boundary checks include:

- required grammar columns are present;
- no columns outside grammar after finalization;
- entity/date or entity/date/event keys are unique;
- stable ID columns map consistently to their source string columns;
- target bucket columns exist for every selected numeric target.

## Environment

**Use the Bash tool (MINGW64 / Git Bash) to run Python scripts.** The Claude
Code PowerShell sandbox blocks conda DLL loading and returns exit 127 for any
script that imports conda packages. The Bash tool works after the conda DLL
paths are added to PATH.

Python is in conda env **enterprise**:

```
/c/ProgramData/miniconda3/envs/enterprise/python.exe
```

Every Bash session must set PATH before running Python:

```bash
CONDA_ENV='/c/ProgramData/miniconda3/envs/enterprise'
export PATH="$CONDA_ENV:$CONDA_ENV/Library/bin:$CONDA_ENV/Library/usr/bin:$CONDA_ENV/Library/mingw-w64/bin:$CONDA_ENV/Scripts:$PATH"
PY="$CONDA_ENV/python.exe"
```

API keys are in `api_keys.sh` as bash `export` statements. Source it:

```bash
source /c/Users/dotep/Documents/unified_wager/api_keys.sh
```

Required env vars:

- `ALPHA_VANTAGE_API_KEY` — used by `av_loader.py`
- `FRED_API_KEY` — used by `fred_loader.py`
- `FOOTYSTATS_API_KEY` — used by `footy_loader.py`
- `KENPOM_API` — used by `ncaa_loader.py`

## Market Pipeline — End-to-End Example

Run these from the Bash tool. Include the session setup block at the start.

```bash
# --- session setup ---
source /c/Users/dotep/Documents/unified_wager/api_keys.sh
CONDA_ENV='/c/ProgramData/miniconda3/envs/enterprise'
export PATH="$CONDA_ENV:$CONDA_ENV/Library/bin:$CONDA_ENV/Library/usr/bin:$CONDA_ENV/Library/mingw-w64/bin:$CONDA_ENV/Scripts:$PATH"
PY="$CONDA_ENV/python.exe"
cd /c/Users/dotep/Documents/unified_wager

# 1. Download Alpha Vantage daily-adjusted prices (one .pqt per symbol)
"$PY" av_loader.py -s ./data/av --symbols SPY QQQ AAPL MSFT

# 2. Download FRED macro series (one .pqt per series id)
"$PY" fred_loader.py -s ./data/fred --series_ids DFF T10Y2Y UNRATE

# 3. Build the feature + target dataset
"$PY" market_dataset_builder.py \
  --av_dir ./data/av \
  --fred_dir ./data/fred \
  --save_to ./data/market.pqt \
  --symbols AAPL MSFT \
  --anchors SPY DFF T10Y2Y \
  --relative_symbol SPY \
  --forward_returns 5 21 \
  --positive_bucket_count 9

# 4a. Train LightGBM bucket probability models (market_predictor.py)
"$PY" market_predictor.py \
  --dataset ./data/market.pqt \
  --save_to ./models \
  --predict_to ./data/market_predictions.pqt

# 4b. OR run the generic sklearn poc model (any grammar dataset)
"$PY" poc_model.py \
  --data ./data/market.pqt \
  --save_to ./models \
  --target TARGET_EXCESS_LOG_RETURN_5
```

`market_predictor.py` writes one `model_{suffix}.lgb` and `model_{suffix}.json`
per target period to `--save_to`. The JSON sidecar records the input column list
and best iteration for reproducible inference.

## Common Commands

Show script arguments:

```powershell
python av_loader.py --help
python fred_loader.py --help
python market_dataset_builder.py --help
python market_predictor.py --help
python poc_model.py --help
python footy_loader.py --help
python footy_dataset_builder.py --help
python tennis_loader.py --help
python matchstat_loader.py --help
python ncaa_loader.py --help
```

Compile-check the Python files:

```bash
"$PY" -B -m compileall *.py
```

## Assistant Checklist

Before changing code, read `CONVENTIONS.md` and the relevant module. Preserve
the loader/builder separation. Keep new columns inside the grammar. Prefer
existing utilities over new helper abstractions. Validate outputs where data
crosses a boundary. Avoid secrets in code, logs, or docs.
