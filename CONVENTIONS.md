# Unified Wager Coding Conventions

## Python Style

- Add type hints for everything.
- Use lowercase, single-line, future-friendly docstrings.
- Keep nesting to 2 indents max, and prefer 1 when possible.
- Use `_` only for intentionally unused variables.
- Prefer comprehensions over loops when they stay readable.
- Prefer positive-case conditionals first.
- Prefer efficient single-line conditionals when they improve clarity.
- Use nested helpers only when they are necessary or materially improve readability.
- Prefer first-class top-level helpers over clever wrappers around repetitive params.
- Prefer standard Python and pandas over custom helper abstractions when the standard approach is clear.
- Honor pylint limits for method line count and parameter count.
- Use dataclasses for repeated structured config, inputs, and return shapes.
- Minimize defensive coding; prefer failing fast over silently masking unexpected internal states.
- Avoid redundant casting when a value is already typed, such as `y = int(x)` inside `def method(x: int)`.
- Do not wrap basic Python or pandas operations in trivial helpers just to rename them.
- Use small shared constants for repeated keys when they define a shared contract or improve consistency.
- Keep one-off literals and one-method-only strings local to the method that owns them.
- Hoist values into config, constants, or helpers only when they are genuinely reused, define a shared contract, or materially improve readability.

## Data Grammar

Every normalized loader output should use the engine grammar:

- `LABEL_`: human-readable metadata and row context, especially dates and source labels.
- `SERIES_`: identity columns for the modeled entity or series, such as `SERIES_SYMBOL`, `SERIES_ID`, `SERIES_HOME`, or `SERIES_AWAY`.
- `FEATURE_`: continuous model inputs available at prediction time.
- `TARGET_`: values selected by later feature or modeling layers for direct prediction or distribution learning.
- `IS_`: binary flags.
- `CATEGORY_`: categorical variables.

Prefer `LABEL_DATE` as the canonical time column unless a dataset genuinely needs a more specific label.

Use prediction-time availability in feature-building and modeling layers as the deciding rule between `FEATURE_` and `TARGET_`. Loader outputs should not choose targets; keep raw observed numeric fields under `FEATURE_` until a later layer defines a prediction task.

Use stable unsigned 64-bit integer IDs for string/object `SERIES_` columns when they speed joins, grouping, or modeling. Derive them as `SERIES_<NAME>_ID` from the normalized string/object source value. IDs must be deterministic across runs and independent of which other series are present, so partitioned per-series files can be regenerated without changing IDs. When a provider also supplies IDs, preserve them as `LABEL_SOURCE_*_ID` so `SERIES_*_ID` remains the custom fast integer form.

## Loader Architecture

- Loaders produce raw, grammar-normalized observations.
- Loaders should emit `FEATURE_` rather than `TARGET_`; target selection belongs in a later feature-building or labeling stage.
- Loaders may append stable custom unsigned integer `SERIES_*_ID` columns for normalized string/object series columns.
- Loaders should validate the grammar at save boundaries.
- API loaders should resolve keys in one order: `--api_key`, then `--api_key_file`, then the provider environment variable.
- Loader `AppConfig` classes should inherit `BaseAppConfig` and initialize any additional directories they rely on.
- Loader parsers should use shared parser helpers for standard `--save_to`, `--api_key`, and `--api_key_file` arguments.
- Do not hardcode API keys or secrets in Python modules.
- Provider environment variable names should match `api_keys.sh`.
- Loaders should handle expected external failures gracefully, such as bad remote payloads, rate limits, and per-symbol download misses.
- Internal contract violations should fail fast.
- Validate at pipeline boundaries, then trust those contracts internally.
- Shared request, logging, download, schema, and grammar utilities should live in one common utility module.

Do not put derivative feature construction in loaders. This includes:

- Rolling, expanding, lagged, shifted, or EWM values.
- Historical aggregations and medians.
- Imputation waterfalls or imputation status features.
- Odds-derived edges or model prediction joins.
- Ratios, deltas, rankings, momentum, form, or trend features derived from history.
- Bucket assignments for target distributions.
- API-side transformations or aggregations when raw values are available.

Derivative features belong in later feature-building stages.

## Prediction Engine Direction

The engine should be dataset-agnostic. It should consume grammar-normalized time-series records and learn:

- A raw regression prediction for a target value.
- A probabilistic distribution over the target value.
- Smart target buckets based on historical distributions.

The loader layer should make heterogeneous datasets legible to that engine without deciding model-specific feature formulations too early.
