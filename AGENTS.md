# Project Instructions

Follow the conventions in `CONVENTIONS.md` for this project.

Key reminders:

- Use the normalized data grammar: `LABEL_`, `SERIES_`, `FEATURE_`, `TARGET_`, `IS_`, and `CATEGORY_`.
- Keep loaders raw-only and grammar-normalized.
- Keep loader output target-free; define `TARGET_` columns in later feature-building or labeling stages.
- Add stable custom unsigned integer `SERIES_*_ID` columns for loader string/object series columns when useful.
- Put derivative data formulations in later feature-building stages, not loaders.
- Validate contracts at pipeline boundaries and fail fast on internal contract violations.
