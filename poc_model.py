from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_building import make_column_token, target_suffix
from market_data import LABEL_DATE
from utilities import load_parquet, make_save_to_parser, save_parquet, write_json


@dataclass(frozen=True, slots=True)
class AppConfig:
    """hold runtime settings"""

    data: Path
    save_to: Path
    target: str
    test_fraction: float = 0.2
    max_rows: int = 0
    max_category_levels: int = 100
    include_series: bool = False
    random_state: int = 42
    n_estimators: int = 300
    min_samples_leaf: int = 5

    def __post_init__(self) -> None:
        if not self.data.exists():
            raise ValueError(f"data does not exist: {self.data}")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1")
        if self.max_rows < 0:
            raise ValueError("max_rows must be non-negative")
        if self.max_category_levels <= 0:
            raise ValueError("max_category_levels must be positive")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        if self.target and not self.target.startswith("TARGET_"):
            raise ValueError(f"target must start with TARGET_: {self.target}")
        self.save_to.mkdir(parents=True, exist_ok=True)

    @property
    def metrics_path(self) -> Path:
        """return metrics output path"""
        return self.save_to / "poc_model_metrics.json"

    @property
    def predictions_path(self) -> Path:
        """return prediction output path"""
        return self.save_to / "poc_model_predictions.pqt"

    @property
    def model_path(self) -> Path:
        """return model output path"""
        return self.save_to / "poc_model.joblib"


@dataclass(frozen=True, slots=True)
class FeatureColumns:
    """hold selected model feature columns"""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def all_columns(self) -> list[str]:
        """return all selected columns"""
        return list(self.numeric + self.categorical)


@dataclass(frozen=True, slots=True)
class BucketTarget:
    """hold target bucket metadata"""

    type_column: str
    class_column: str
    bucket_columns: tuple[str, ...]

    @property
    def exists(self) -> bool:
        """return whether bucket target columns are available"""
        return bool(self.type_column or self.bucket_columns)


@dataclass(frozen=True, slots=True)
class SplitFrames:
    """hold train and test frames"""

    train: pd.DataFrame
    test: pd.DataFrame


def parse_arguments() -> Namespace:
    """parse cli arguments"""
    parser: ArgumentParser = make_save_to_parser(
        description="train a grammar-normalized poc wagering model",
        save_to_help="output directory for model artifacts",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="grammar-normalized parquet dataset with TARGET_ columns",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="numeric TARGET_ column to regress; inferred only if unambiguous",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="held-out fraction; chronological when LABEL_DATE exists",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="optional maximum rows after target filtering; 0 keeps all rows",
    )
    parser.add_argument(
        "--max_category_levels",
        type=int,
        default=100,
        help="drop categorical columns above this cardinality",
    )
    parser.add_argument(
        "--include_series",
        action="store_true",
        help="include SERIES_ columns as categorical model inputs",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="random seed for split and forest models",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=300,
        help="number of trees for each random forest head",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=5,
        help="minimum samples per random forest leaf",
    )
    return parser.parse_args()


def build_app_config(args: Namespace) -> AppConfig:
    """build the app config"""
    return AppConfig(
        data=args.data,
        save_to=args.save_to,
        target=args.target,
        test_fraction=args.test_fraction,
        max_rows=args.max_rows,
        max_category_levels=args.max_category_levels,
        include_series=args.include_series,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
    )


def is_bucket_metadata(column: str) -> bool:
    """return whether a target column is bucket metadata"""
    return column.startswith(
        ("TARGET_TYPE_", "TARGET_CLASS_", "TARGET_BUCKET_")
    )


def numeric_target_candidates(df: pd.DataFrame) -> list[str]:
    """return numeric regression target candidates"""
    return [
        column
        for column in df.columns
        if column.startswith("TARGET_")
        and not is_bucket_metadata(column)
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def resolve_target(df: pd.DataFrame, requested_target: str) -> str:
    """resolve the selected regression target"""
    if requested_target:
        if requested_target not in df.columns:
            raise ValueError(f"target column not found: {requested_target}")
        if not pd.api.types.is_numeric_dtype(df[requested_target]):
            raise ValueError(f"target must be numeric: {requested_target}")
        if is_bucket_metadata(requested_target):
            raise ValueError(f"target cannot be bucket metadata: {requested_target}")
        return requested_target

    candidates: list[str] = numeric_target_candidates(df)
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "provide --target because numeric TARGET_ candidates are "
        f"{candidates}"
    )


def bucket_target_for(df: pd.DataFrame, target: str) -> BucketTarget:
    """find bucket columns for a selected target"""
    suffix: str = target_suffix(target)
    type_column: str = f"TARGET_TYPE_{suffix}"
    class_column: str = f"TARGET_CLASS_{suffix}"
    bucket_prefix: str = f"TARGET_BUCKET_{suffix}_"
    return BucketTarget(
        type_column=type_column if type_column in df.columns else "",
        class_column=class_column if class_column in df.columns else "",
        bucket_columns=tuple(
            column for column in df.columns if column.startswith(bucket_prefix)
        ),
    )


def prepare_training_frame(
    df: pd.DataFrame,
    target: str,
    config: AppConfig,
) -> pd.DataFrame:
    """filter rows to trainable target observations"""
    output: pd.DataFrame = df.copy()
    output[target] = pd.to_numeric(output[target], errors="coerce")
    output = output.loc[np.isfinite(output[target].to_numpy(dtype="float64"))]
    if output.empty:
        raise ValueError(f"no rows with finite target values: {target}")

    if LABEL_DATE in output.columns:
        output = output.sort_values(LABEL_DATE, kind="stable")
        if config.max_rows:
            return output.tail(config.max_rows).reset_index(drop=True)
        return output.reset_index(drop=True)

    if config.max_rows and len(output) > config.max_rows:
        return output.sample(
            n=config.max_rows,
            random_state=config.random_state,
        ).reset_index(drop=True)
    return output.reset_index(drop=True)


def split_frame(df: pd.DataFrame, config: AppConfig) -> SplitFrames:
    """split frame chronologically or randomly"""
    if len(df) < 5:
        raise ValueError("at least 5 target rows are required for the poc split")
    test_rows: int = max(1, int(round(len(df) * config.test_fraction)))
    train_rows: int = len(df) - test_rows
    if train_rows < 2:
        raise ValueError("not enough rows left for training after split")

    if LABEL_DATE in df.columns:
        return SplitFrames(
            train=df.iloc[:train_rows].copy(),
            test=df.iloc[train_rows:].copy(),
        )

    shuffled: pd.DataFrame = df.sample(
        frac=1.0,
        random_state=config.random_state,
    ).reset_index(drop=True)
    return SplitFrames(
        train=shuffled.iloc[:train_rows].copy(),
        test=shuffled.iloc[train_rows:].copy(),
    )


def categorical_candidates(df: pd.DataFrame, config: AppConfig) -> list[str]:
    """return categorical feature candidates"""
    prefixes: tuple[str, ...] = ("CATEGORY_",)
    if config.include_series:
        prefixes = ("CATEGORY_", "SERIES_")
    return [
        column
        for column in df.columns
        if column.startswith(prefixes) and not column.startswith("TARGET_")
    ]


def select_feature_columns(df: pd.DataFrame, config: AppConfig) -> FeatureColumns:
    """select grammar-compatible model inputs"""
    numeric_columns: tuple[str, ...] = tuple(
        column
        for column in df.columns
        if column.startswith(("FEATURE_", "IS_"))
        and pd.api.types.is_numeric_dtype(df[column])
    )
    categorical_columns: tuple[str, ...] = tuple(
        column
        for column in categorical_candidates(df, config)
        if df[column].nunique(dropna=True) <= config.max_category_levels
    )
    features: FeatureColumns = FeatureColumns(
        numeric=numeric_columns,
        categorical=categorical_columns,
    )
    if not features.all_columns:
        raise ValueError("no usable FEATURE_, IS_, CATEGORY_, or SERIES_ inputs found")
    return features


def make_one_hot_encoder() -> OneHotEncoder:
    """make a version-compatible one-hot encoder"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_preprocessor(features: FeatureColumns) -> ColumnTransformer:
    """build the feature preprocessor"""
    transformers: list[tuple[str, Pipeline | OneHotEncoder, list[str]]] = []
    if features.numeric:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                list(features.numeric),
            )
        )
    if features.categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                list(features.categorical),
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_regressor(config: AppConfig, features: FeatureColumns) -> Pipeline:
    """build the regression pipeline"""
    return Pipeline(
        [
            ("preprocess", make_preprocessor(features)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=config.n_estimators,
                    min_samples_leaf=config.min_samples_leaf,
                    n_jobs=-1,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def make_classifier(config: AppConfig, features: FeatureColumns) -> Pipeline:
    """build the bucket classification pipeline"""
    return Pipeline(
        [
            ("preprocess", make_preprocessor(features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    min_samples_leaf=config.min_samples_leaf,
                    n_jobs=-1,
                    random_state=config.random_state,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def bucket_series(df: pd.DataFrame, bucket_target: BucketTarget) -> pd.Series:
    """resolve a class id series from bucket target columns"""
    if bucket_target.type_column:
        values: pd.Series = pd.to_numeric(
            df[bucket_target.type_column],
            errors="coerce",
        )
        return values.where(values.ge(0)).astype("Int64")

    bucket_values: pd.DataFrame = df.loc[:, list(bucket_target.bucket_columns)]
    valid_mask: pd.Series = bucket_values.sum(axis=1).eq(1)
    class_ids: pd.Series = pd.Series(pd.NA, index=df.index, dtype="Int64")
    class_ids.loc[valid_mask] = bucket_values.loc[valid_mask].to_numpy().argmax(axis=1)
    return class_ids


def bucket_class_lookup(
    df: pd.DataFrame,
    bucket_target: BucketTarget,
) -> dict[int, str]:
    """map bucket type ids to class labels"""
    if bucket_target.type_column and bucket_target.class_column:
        pairs: pd.DataFrame = df.loc[
            df[bucket_target.type_column].notna()
            & df[bucket_target.class_column].notna(),
            [bucket_target.type_column, bucket_target.class_column],
        ].drop_duplicates()
        return {
            int(row[bucket_target.type_column]): str(row[bucket_target.class_column])
            for _, row in pairs.iterrows()
        }
    return {
        idx: column.removeprefix("TARGET_BUCKET_")
        for idx, column in enumerate(bucket_target.bucket_columns)
    }


def rmse(actual: pd.Series, predicted: np.ndarray) -> float:
    """compute root mean squared error"""
    return float(mean_squared_error(actual, predicted) ** 0.5)


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """compute regression metrics"""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": rmse(actual, predicted),
        "r2": float(r2_score(actual, predicted)),
    }


def classification_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    """compute classification metrics"""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(actual, predicted)),
    }
    if len(classes) > 1:
        metrics["log_loss"] = float(
            log_loss(actual, probabilities, labels=list(classes))
        )
    return metrics


def probability_column_name(target: str, class_name: str) -> str:
    """build a probability output column name"""
    return f"TARGET_PROB_{target_suffix(target)}_{make_column_token(class_name)}"


def prediction_context_columns(df: pd.DataFrame) -> list[str]:
    """select context columns for prediction output"""
    return [
        column
        for column in df.columns
        if column.startswith(("LABEL_", "SERIES_"))
    ]


def build_prediction_frame(
    test_df: pd.DataFrame,
    target: str,
    regression_predictions: np.ndarray,
    classifier: Pipeline | None,
    bucket_target: BucketTarget,
) -> pd.DataFrame:
    """build the saved prediction artifact"""
    suffix: str = target_suffix(target)
    output: pd.DataFrame = test_df.loc[:, prediction_context_columns(test_df)].copy()
    output[f"TARGET_ACTUAL_{suffix}"] = test_df[target].to_numpy()
    output[f"TARGET_PRED_{suffix}"] = regression_predictions
    if classifier is None:
        return output

    probabilities: np.ndarray = classifier.predict_proba(test_df)
    predictions: np.ndarray = classifier.predict(test_df)
    lookup: dict[int, str] = bucket_class_lookup(test_df, bucket_target)
    output[f"TARGET_PRED_CLASS_{suffix}"] = [
        lookup.get(int(value), str(value)) for value in predictions
    ]
    for idx, class_value in enumerate(classifier.named_steps["model"].classes_):
        class_name: str = lookup.get(int(class_value), str(class_value))
        output[probability_column_name(target, class_name)] = probabilities[:, idx]
    return output


def train_classifier(
    splits: SplitFrames,
    features: FeatureColumns,
    bucket_target: BucketTarget,
    config: AppConfig,
) -> tuple[Pipeline | None, dict[str, Any]]:
    """train the bucket classifier when bucket targets exist"""
    if not bucket_target.exists:
        return None, {"status": "skipped", "reason": "no bucket target columns"}

    train_classes: pd.Series = bucket_series(splits.train, bucket_target)
    test_classes: pd.Series = bucket_series(splits.test, bucket_target)
    train_mask: pd.Series = train_classes.notna()
    test_mask: pd.Series = test_classes.notna()
    if train_classes.loc[train_mask].nunique() < 2:
        return None, {"status": "skipped", "reason": "fewer than two classes"}
    if int(test_mask.sum()) == 0:
        return None, {"status": "skipped", "reason": "no test rows with classes"}

    classifier: Pipeline = make_classifier(config, features)
    classifier.fit(
        splits.train.loc[train_mask],
        train_classes.loc[train_mask].astype(int),
    )
    predicted: np.ndarray = classifier.predict(splits.test.loc[test_mask])
    probabilities: np.ndarray = classifier.predict_proba(splits.test.loc[test_mask])
    classes: np.ndarray = classifier.named_steps["model"].classes_
    metrics: dict[str, Any] = {
        "status": "trained",
        "target_type_column": bucket_target.type_column,
        "bucket_columns": list(bucket_target.bucket_columns),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "classes": [int(value) for value in classes],
    }
    metrics.update(
        classification_metrics(
            test_classes.loc[test_mask].astype(int),
            predicted,
            probabilities,
            classes,
        )
    )
    return classifier, metrics


def save_model_artifact(
    config: AppConfig,
    regressor: Pipeline,
    classifier: Pipeline | None,
    features: FeatureColumns,
    target: str,
) -> None:
    """save fitted model pipelines"""
    import joblib

    joblib.dump(
        {
            "target": target,
            "features": asdict(features),
            "config": asdict(config),
            "regressor": regressor,
            "classifier": classifier,
        },
        config.model_path,
    )


def main() -> None:
    """train the poc model"""
    config: AppConfig = build_app_config(parse_arguments())
    raw_df: pd.DataFrame = load_parquet(config.data)
    target: str = resolve_target(raw_df, config.target)
    df: pd.DataFrame = prepare_training_frame(raw_df, target, config)
    splits: SplitFrames = split_frame(df, config)
    features: FeatureColumns = select_feature_columns(splits.train, config)
    bucket_target: BucketTarget = bucket_target_for(df, target)

    regressor: Pipeline = make_regressor(config, features)
    regressor.fit(splits.train, splits.train[target])
    regression_predictions: np.ndarray = regressor.predict(splits.test)

    classifier, bucket_metrics = train_classifier(
        splits,
        features,
        bucket_target,
        config,
    )
    prediction_df: pd.DataFrame = build_prediction_frame(
        splits.test,
        target,
        regression_predictions,
        classifier,
        bucket_target,
    )
    save_parquet(prediction_df, config.predictions_path)
    save_model_artifact(config, regressor, classifier, features, target)

    metrics: dict[str, Any] = {
        "target": target,
        "data": str(config.data),
        "train_rows": len(splits.train),
        "test_rows": len(splits.test),
        "features": asdict(features),
        "regression": regression_metrics(splits.test[target], regression_predictions),
        "bucket_classifier": bucket_metrics,
        "artifacts": {
            "metrics": str(config.metrics_path),
            "predictions": str(config.predictions_path),
            "model": str(config.model_path),
        },
    }
    write_json(str(config.metrics_path), metrics)


if __name__ == "__main__":
    main()
