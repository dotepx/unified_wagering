# Unified Wager — Claude Code Context

Full docs: `LLM_PROJECT_GUIDE.md` | Style rules: `CONVENTIONS.md`

## Environment

**Use the Bash tool, not PowerShell**, to run Python scripts. The Claude Code
PowerShell sandbox blocks native DLL loading for conda packages (exit 127).

### CPU work — MINGW64 (Git Bash, Bash tool default)

Python at `/c/ProgramData/miniconda3/envs/enterprise/python.exe`. Must prepend
conda DLL paths every session:

```bash
CONDA_ENV='/c/ProgramData/miniconda3/envs/enterprise'
export PATH="$CONDA_ENV:$CONDA_ENV/Library/bin:$CONDA_ENV/Library/usr/bin:$CONDA_ENV/Library/mingw-w64/bin:$CONDA_ENV/Scripts:$PATH"
PY="$CONDA_ENV/python.exe"
```

### GPU work — WSL (TensorFlow + CUDA configured here)

WSL Python: `/home/dotepx/.conda/envs/enterprise/bin/python`

Invoke from the Bash tool via `wsl`:

```bash
wsl -- bash -lc "
  source /mnt/c/Users/dotep/Documents/unified_wager/api_keys.sh
  cd /mnt/c/Users/dotep/Documents/unified_wager
  /home/dotepx/.conda/envs/enterprise/bin/python market_sequence_predictor.py \
    --dataset ./data/market.pqt \
    --save_to ./cnn_models \
    --predict_to ./data/market_cnn_predictions.pqt
"
```

Use WSL for any TensorFlow training that should use the GPU.

### API keys

In `api_keys.sh` (bash `export` format). Source it at the top of every session:

```bash
source /c/Users/dotep/Documents/unified_wager/api_keys.sh   # MINGW64
# or
source /mnt/c/Users/dotep/Documents/unified_wager/api_keys.sh  # WSL
```

## Data Grammar (memorize these prefixes)

| Prefix | Role |
|---|---|
| `LABEL_` | Row metadata / dates |
| `SERIES_` | Entity identity (`SERIES_SYMBOL`, `SERIES_HOME`, …) |
| `FEATURE_` | Numeric model inputs available at prediction time |
| `TARGET_` | Forward-looking labels and bucket metadata |
| `CATEGORY_` | Categorical variables (year, month, week) |
| `IS_` | Binary flags |

`TARGET_` sub-types: numeric regression (`TARGET_EXCESS_LOG_RETURN_5`),
bucket id (`TARGET_TYPE_*`), class label (`TARGET_CLASS_*`),
one-hot indicator (`TARGET_BUCKET_*_NN`).

## Market Pipeline — Working Commands

Run these from the Bash tool. Set PATH and API keys at the top of every session.

```bash
# --- session setup (required every Bash session) ---
source /c/Users/dotep/Documents/unified_wager/api_keys.sh
CONDA_ENV='/c/ProgramData/miniconda3/envs/enterprise'
export PATH="$CONDA_ENV:$CONDA_ENV/Library/bin:$CONDA_ENV/Library/usr/bin:$CONDA_ENV/Library/mingw-w64/bin:$CONDA_ENV/Scripts:$PATH"
PY="$CONDA_ENV/python.exe"
cd /c/Users/dotep/Documents/unified_wager

# 1. Download AV daily-adjusted prices (one .pqt per symbol)
"$PY" av_loader.py -s ./data/av --symbols SPY QQQ AAPL MSFT

# 2. Download FRED macro series (one .pqt per series)
"$PY" fred_loader.py -s ./data/fred --series_ids DFF T10Y2Y UNRATE

# 3. Build feature + target dataset
"$PY" market_dataset_builder.py \
  --av_dir ./data/av \
  --fred_dir ./data/fred \
  --save_to ./data/market.pqt \
  --symbols AAPL MSFT \
  --anchors SPY DFF T10Y2Y \
  --relative_symbol SPY \
  --forward_returns 5 21 \
  --positive_bucket_count 9

# 4a. Train LightGBM bucket predictor (pointwise, market_predictor.py)
"$PY" market_predictor.py \
  --dataset ./data/market.pqt \
  --save_to ./models \
  --predict_to ./data/market_predictions.pqt

# 4b. Train residual 1D CNN predictor (sequence-based, market_sequence_predictor.py)
"$PY" market_sequence_predictor.py \
  --dataset ./data/market.pqt \
  --save_to ./cnn_models \
  --predict_to ./data/market_cnn_predictions.pqt \
  --seq_len 60

# 4b. Train sklearn poc model (generic, any grammar dataset)
"$PY" poc_model.py \
  --data ./data/market.pqt \
  --save_to ./models \
  --target TARGET_EXCESS_LOG_RETURN_5
```

## Module Map (one line each)

| File | What it does |
|---|---|
| `utilities.py` | Shared CLI, IO, grammar, ID, logging helpers |
| `market_data.py` | Market column constants |
| `feature_building.py` | Log ratios, EWM features, calendar cats, quantile buckets |
| `av_loader.py` | Alpha Vantage → per-symbol `.pqt` (raw prices) |
| `fred_loader.py` | FRED API → per-series `.pqt` (macro observations) |
| `market_dataset_builder.py` | AV + FRED → feature/target dataset `.pqt` |
| `market_predictor.py` | Dataset `.pqt` → LightGBM bucket probability models (pointwise) |
| `market_sequence_predictor.py` | Dataset `.pqt` → residual 1D CNN bucket models (sequence-based, TensorFlow) |
| `poc_model.py` | Dataset `.pqt` → sklearn RF regression + bucket classifier |
| `footy_loader.py` | FootyStats API → football match `.pqt` |
| `footy_dataset_builder.py` | Football `.pqt` → feature/target dataset |
| `tennis_loader.py` | Tennis-data.co.uk → match `.pqt` |
| `matchstat_loader.py` | MatchStat/RapidAPI fixtures → `.pqt` |
| `ncaa_loader.py` | KenPom → NCAA basketball `.pqt` |

## Layering Rules (never break these)

- **Loaders**: raw fetch + grammar normalize + validate + save. No `TARGET_` columns. No rolling/lagged/derivative features.
- **Dataset builders**: add features, define `TARGET_` columns, add buckets via `add_positive_quantile_target_buckets`, call `smart_type` + `finalize_grammar_frame` before saving.
- **Model code**: reads `FEATURE_*` / `CATEGORY_*` as inputs; selects `TARGET_TYPE_*` as classification labels; is domain-agnostic.

## Compile Check

```bash
"$PY" -B -m compileall *.py
```
