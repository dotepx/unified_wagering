from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from market_data import LABEL_DATE
from utilities import load_parquet, save_parquet, update_status


__all__: list[str] = []

if __name__ not in {"__main__", "__mp_main__"}:
    raise ImportError(
        "market_predictor.py is a script entrypoint and should not be imported"
    )


TARGET_TYPE_PREFIX: str = "TARGET_TYPE_"
PRED_BUCKET_PREFIX: str = "PRED_BUCKET_"
INVALID_BUCKET: int = -1


@dataclass(frozen=True, slots=True)
class AppConfig:
    """hold runtime settings"""

    dataset: Path
    save_to: Path
    predict_to: Path | None
    test_fraction: float
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    early_stopping_rounds: int
    seed: int

    def __post_init__(self) -> None:
        if not self.dataset.exists():
            raise ValueError(f"dataset does not exist: {self.dataset}")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError(
                f"test_fraction must be between 0 and 1: {self.test_fraction}"
            )
        self.save_to.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """hold one target column spec"""

    column: str
    suffix: str
    bucket_count: int


def parse_arguments() -> Namespace:
    """parse cli arguments"""
    parser: ArgumentParser = ArgumentParser(
        description="train lgbm bucket probability predictors from a market dataset"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="market dataset parquet path (output of market_dataset_builder)",
    )
    parser.add_argument(
        "--save_to",
        type=Path,
        required=True,
        help="directory to write trained model and metadata files",
    )
    parser.add_argument(
        "--predict_to",
        type=Path,
        default=None,
        help="optional output parquet path for bucket probability predictions",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="fraction of time-sorted dates held out for evaluation",
    )
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=64)
    parser.add_argument("--min_data_in_leaf", type=int, default=20)
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_app_config(args: Namespace) -> AppConfig:
    """build runtime config from parsed arguments"""
    return AppConfig(
        dataset=args.dataset,
        save_to=args.save_to,
        predict_to=args.predict_to,
        test_fraction=args.test_fraction,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
    )


def discover_target_specs(df: pd.DataFrame) -> list[TargetSpec]:
    """discover TARGET_TYPE_* columns and derive their specs"""
    specs: list[TargetSpec] = []
    for column in df.columns:
        if not column.startswith(TARGET_TYPE_PREFIX):
            continue
        suffix: str = column.removeprefix(TARGET_TYPE_PREFIX)
        valid: np.ndarray = np.sort(
            df.loc[df[column].ne(INVALID_BUCKET), column].unique().astype(int)
        )
        if valid.size == 0:
            continue
        specs.append(
            TargetSpec(
                column=column,
                suffix=suffix,
                bucket_count=int(valid.max()) + 1,
            )
        )
    return sorted(specs, key=lambda s: s.column)


def select_input_columns(df: pd.DataFrame) -> list[str]:
    """return all FEATURE_* and CATEGORY_* columns as model inputs"""
    return [
        col
        for col in df.columns
        if col.startswith("FEATURE_") or col.startswith("CATEGORY_")
    ]


def date_split(
    df: pd.DataFrame,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """split on unique dates to avoid cross-symbol leakage"""
    sorted_dates: np.ndarray = np.sort(df[LABEL_DATE].unique())
    split_idx: int = int(len(sorted_dates) * (1.0 - test_fraction))
    split_date = sorted_dates[split_idx]
    train: pd.DataFrame = df.loc[df[LABEL_DATE].lt(split_date)].copy()
    test: pd.DataFrame = df.loc[df[LABEL_DATE].ge(split_date)].copy()
    return train, test


def cross_entropy_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """compute mean cross-entropy over predicted class probabilities"""
    n: int = len(y_true)
    clipped: np.ndarray = np.clip(y_proba[np.arange(n), y_true], 1e-15, 1.0)
    return float(-np.log(clipped).mean())


def train_one_target(
    df: pd.DataFrame,
    input_columns: list[str],
    spec: TargetSpec,
    config: AppConfig,
) -> lgb.LGBMClassifier:
    """train one lgbm multiclass model for a single target spec"""
    valid: pd.DataFrame = df.loc[df[spec.column].ne(INVALID_BUCKET)].copy()
    if valid.empty:
        raise ValueError(f"no valid rows for target {spec.column}")

    train_df, test_df = date_split(valid, config.test_fraction)
    x_train: pd.DataFrame = train_df[input_columns]
    y_train: pd.Series = train_df[spec.column].astype(int)
    x_test: pd.DataFrame = test_df[input_columns]
    y_test: pd.Series = test_df[spec.column].astype(int)

    update_status(
        f"training {spec.column}: {len(x_train)} train rows, "
        f"{len(x_test)} test rows, {spec.bucket_count} buckets"
    )

    model: lgb.LGBMClassifier = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=spec.bucket_count,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_data_in_leaf,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        random_state=config.seed,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    test_proba: np.ndarray = model.predict_proba(x_test)
    loss: float = cross_entropy_loss(y_test.to_numpy(), test_proba)
    update_status(
        f"{spec.column}: log_loss={loss:.4f}, "
        f"best_iteration={model.best_iteration_}"
    )
    return model


def save_model_artifacts(
    model: lgb.LGBMClassifier,
    input_columns: list[str],
    spec: TargetSpec,
    save_dir: Path,
) -> None:
    """save model booster and metadata sidecar"""
    model_path: Path = save_dir / f"model_{spec.suffix}.lgb"
    meta_path: Path = save_dir / f"model_{spec.suffix}.json"
    model.booster_.save_model(str(model_path))
    meta: dict[str, object] = {
        "suffix": spec.suffix,
        "target_column": spec.column,
        "bucket_count": spec.bucket_count,
        "best_iteration": model.best_iteration_,
        "input_columns": input_columns,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    update_status(f"saved model to {model_path}")


def add_bucket_predictions(
    df: pd.DataFrame,
    input_columns: list[str],
    model: lgb.LGBMClassifier,
    spec: TargetSpec,
) -> pd.DataFrame:
    """append predicted bucket probability columns for one target spec"""
    proba: np.ndarray = model.predict_proba(df[input_columns])
    output: pd.DataFrame = df.copy()
    for bucket_idx in range(spec.bucket_count):
        col: str = f"{PRED_BUCKET_PREFIX}{spec.suffix}_{bucket_idx:02d}"
        output[col] = proba[:, bucket_idx].astype("float32")
    return output


def main() -> None:
    """train bucket probability predictors from a market dataset"""
    config: AppConfig = build_app_config(parse_arguments())
    update_status(f"loading dataset from {config.dataset}")
    df: pd.DataFrame = load_parquet(config.dataset)

    input_columns: list[str] = select_input_columns(df)
    target_specs: list[TargetSpec] = discover_target_specs(df)

    if not input_columns:
        raise ValueError("no FEATURE_* or CATEGORY_* columns found in dataset")
    if not target_specs:
        raise ValueError("no TARGET_TYPE_* columns found in dataset")

    update_status(
        f"found {len(input_columns)} input columns, {len(target_specs)} target specs"
    )

    trained_models: dict[str, lgb.LGBMClassifier] = {}
    for spec in target_specs:
        model: lgb.LGBMClassifier = train_one_target(df, input_columns, spec, config)
        save_model_artifacts(model, input_columns, spec, config.save_to)
        trained_models[spec.suffix] = model

    if config.predict_to is None:
        return

    update_status(f"generating predictions for {len(df)} rows")
    predictions: pd.DataFrame = df.copy()
    for spec in target_specs:
        predictions = add_bucket_predictions(
            predictions, input_columns, trained_models[spec.suffix], spec
        )

    save_parquet(predictions, config.predict_to)
    update_status(f"saved predictions to {config.predict_to}")


if __name__ == "__main__":
    main()
