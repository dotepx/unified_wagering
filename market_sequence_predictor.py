from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from market_data import LABEL_DATE, SERIES_SYMBOL_ID
from utilities import load_parquet, save_parquet, update_status


__all__: list[str] = []

if __name__ not in {"__main__", "__mp_main__"}:
    raise ImportError(
        "market_sequence_predictor.py is a script entrypoint and should not be imported"
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
    seq_len: int
    filters: tuple[int, ...]
    kernel_size: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    test_fraction: float
    seed: int

    def __post_init__(self) -> None:
        if not self.dataset.exists():
            raise ValueError(f"dataset does not exist: {self.dataset}")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError(
                f"test_fraction must be between 0 and 1: {self.test_fraction}"
            )
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive: {self.seq_len}")
        if not self.filters:
            raise ValueError("filters must not be empty")
        self.save_to.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """hold one target column spec"""

    column: str
    suffix: str
    bucket_count: int


@dataclass(frozen=True, slots=True)
class SequenceArrays:
    """hold built sequence arrays and their original row positions"""

    sequences: np.ndarray    # (n, seq_len, n_features)  float32
    targets: np.ndarray      # (n,)  int32; INVALID_BUCKET for no-target rows
    row_indices: np.ndarray  # (n,)  int64 — label-based index into original df


def parse_arguments() -> Namespace:
    """parse cli arguments"""
    parser = ArgumentParser(
        description="train a residual 1d cnn bucket predictor from a market dataset"
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
        help="directory for .keras model files and .json metadata",
    )
    parser.add_argument(
        "--predict_to",
        type=Path,
        default=None,
        help="optional output parquet path for bucket probability predictions",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=60,
        help="number of prior bars in each input sequence",
    )
    parser.add_argument(
        "--filters",
        type=int,
        nargs="+",
        default=[64, 64, 128, 128, 256],
        help="conv filter widths for each residual block",
    )
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="early stopping patience on val_loss",
    )
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_app_config(args: Namespace) -> AppConfig:
    """build runtime config from parsed arguments"""
    return AppConfig(
        dataset=args.dataset,
        save_to=args.save_to,
        predict_to=args.predict_to,
        seq_len=args.seq_len,
        filters=tuple(args.filters),
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )


def discover_target_specs(df: pd.DataFrame) -> list[TargetSpec]:
    """discover TARGET_TYPE_* columns and infer bucket counts"""
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


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """return FEATURE_* columns only — no manually engineered CATEGORY inputs"""
    return [col for col in df.columns if col.startswith("FEATURE_")]


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    seq_len: int,
    training_only: bool,
) -> SequenceArrays:
    """slide a seq_len window over each symbol's time series.

    training_only=True  — only rows where target != INVALID_BUCKET
    training_only=False — all rows with sufficient prior history
    """
    seqs: list[np.ndarray] = []
    targets_out: list[int] = []
    row_indices_out: list[int] = []

    for _, group in df.groupby(SERIES_SYMBOL_ID, sort=False):
        group_sorted = group.sort_values(LABEL_DATE)
        features = group_sorted[feature_columns].to_numpy(dtype="float32")
        target_vals = group_sorted[target_column].to_numpy()
        orig_idx = group_sorted.index.to_numpy()
        n = len(group_sorted)

        for t in range(seq_len, n):
            bucket = int(target_vals[t])
            if training_only and bucket == INVALID_BUCKET:
                continue
            seqs.append(features[t - seq_len : t].copy())
            targets_out.append(bucket)
            row_indices_out.append(int(orig_idx[t]))

    if not seqs:
        raise ValueError(
            f"no sequences built for {target_column!r} "
            f"(seq_len={seq_len}, training_only={training_only})"
        )

    return SequenceArrays(
        sequences=np.stack(seqs, axis=0),
        targets=np.array(targets_out, dtype=np.int32),
        row_indices=np.array(row_indices_out, dtype=np.int64),
    )


def date_split(
    arrays: SequenceArrays,
    df: pd.DataFrame,
    test_fraction: float,
) -> tuple[SequenceArrays, SequenceArrays]:
    """split by unique target dates to avoid cross-symbol leakage"""
    row_dates = df.loc[arrays.row_indices, LABEL_DATE].to_numpy()
    sorted_dates = np.sort(np.unique(row_dates))
    split_idx = int(len(sorted_dates) * (1.0 - test_fraction))
    split_date = sorted_dates[split_idx]
    train_mask = row_dates < split_date
    test_mask = row_dates >= split_date

    def _subset(mask: np.ndarray) -> SequenceArrays:
        return SequenceArrays(
            sequences=arrays.sequences[mask],
            targets=arrays.targets[mask],
            row_indices=arrays.row_indices[mask],
        )

    return _subset(train_mask), _subset(test_mask)


def fit_normalizer(seqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """compute per-feature mean and std from training sequences, ignoring NaN"""
    flat = seqs.reshape(-1, seqs.shape[-1])
    mean = np.nanmean(flat, axis=0).astype("float32")
    std = np.nanstd(flat, axis=0).astype("float32")
    std = np.where(std < 1e-8, 1.0, std).astype("float32")
    return mean, std


def apply_normalizer(
    seqs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """z-score normalize and replace NaN/inf with 0 (the mean after scaling)"""
    return np.nan_to_num(
        (seqs - mean) / std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype("float32")


def residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
) -> tf.Tensor:
    """two conv layers with batch norm and a skip connection"""
    shortcut = x
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, use_bias=False)(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.ReLU()(x)


def build_model(
    seq_len: int,
    n_features: int,
    n_buckets: int,
    filters: tuple[int, ...],
    kernel_size: int,
    dropout: float,
) -> tf.keras.Model:
    """residual 1d cnn: raw sequence in, bucket softmax out"""
    inputs = tf.keras.Input(shape=(seq_len, n_features), name="sequence_input")

    x = tf.keras.layers.Conv1D(
        filters[0], 1, padding="same", name="input_projection"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for f in filters:
        x = residual_block(x, f, kernel_size)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(max(32, filters[-1] // 2), activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(
        n_buckets, activation="softmax", name="bucket_probs"
    )(x)

    return tf.keras.Model(inputs, outputs, name="market_sequence_cnn")


def train_one_target(
    train_seqs: SequenceArrays,
    test_seqs: SequenceArrays,
    spec: TargetSpec,
    config: AppConfig,
) -> tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    """train one cnn model; returns (model, norm_mean, norm_std)"""
    norm_mean, norm_std = fit_normalizer(train_seqs.sequences)
    x_train = apply_normalizer(train_seqs.sequences, norm_mean, norm_std)
    x_test = apply_normalizer(test_seqs.sequences, norm_mean, norm_std)

    update_status(
        f"training {spec.column}: "
        f"{len(x_train)} train seqs, {len(x_test)} test seqs, "
        f"{spec.bucket_count} buckets, input shape {x_train.shape[1:]}"
    )

    model = build_model(
        seq_len=config.seq_len,
        n_features=x_train.shape[2],
        n_buckets=spec.bucket_count,
        filters=config.filters,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )
    model.summary(print_fn=lambda s: update_status(s))

    steps = max(1, len(x_train) // config.batch_size)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.epochs * steps,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    model_path = config.save_to / f"model_{spec.suffix}.keras"
    model.fit(
        x_train,
        train_seqs.targets,
        validation_data=(x_test, test_seqs.targets),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ],
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, test_seqs.targets, verbose=0)
    update_status(
        f"{spec.column}: val_loss={test_loss:.4f}, val_acc={test_acc:.4f}"
    )
    return model, norm_mean, norm_std


def save_artifacts(
    feature_columns: list[str],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    spec: TargetSpec,
    config: AppConfig,
) -> None:
    """save normalization metadata sidecar (model saved by ModelCheckpoint)"""
    meta_path = config.save_to / f"model_{spec.suffix}.json"
    meta: dict[str, object] = {
        "suffix": spec.suffix,
        "target_column": spec.column,
        "bucket_count": spec.bucket_count,
        "seq_len": config.seq_len,
        "feature_columns": feature_columns,
        "norm_mean": norm_mean.tolist(),
        "norm_std": norm_std.tolist(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    update_status(
        f"saved {config.save_to / f'model_{spec.suffix}.keras'} "
        f"and {meta_path.name}"
    )


def run_predictions(
    arrays: SequenceArrays,
    model: tf.keras.Model,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """return predicted probabilities (n, n_buckets)"""
    x = apply_normalizer(arrays.sequences, norm_mean, norm_std)
    return model.predict(x, batch_size=batch_size, verbose=0)


def add_bucket_predictions(
    df: pd.DataFrame,
    arrays: SequenceArrays,
    proba: np.ndarray,
    spec: TargetSpec,
) -> pd.DataFrame:
    """append PRED_BUCKET_* columns; rows without sequences stay NaN"""
    output = df.copy()
    for bucket_idx in range(spec.bucket_count):
        col = f"{PRED_BUCKET_PREFIX}{spec.suffix}_{bucket_idx:02d}"
        output[col] = np.nan
        output.loc[arrays.row_indices, col] = proba[:, bucket_idx].astype(
            "float32"
        )
    return output


def main() -> None:
    """train residual 1d cnn bucket probability predictors"""
    config = build_app_config(parse_arguments())
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)

    update_status(f"loading dataset from {config.dataset}")
    df = load_parquet(config.dataset)

    feature_columns = select_feature_columns(df)
    target_specs = discover_target_specs(df)

    if not feature_columns:
        raise ValueError("no FEATURE_* columns found in dataset")
    if not target_specs:
        raise ValueError("no TARGET_TYPE_* columns found in dataset")

    update_status(
        f"found {len(feature_columns)} feature columns, "
        f"{len(target_specs)} target specs, seq_len={config.seq_len}"
    )

    trained: dict[str, tuple[tf.keras.Model, np.ndarray, np.ndarray]] = {}

    for spec in target_specs:
        arrays = build_sequences(
            df, feature_columns, spec.column, config.seq_len, training_only=True
        )
        train_seqs, test_seqs = date_split(arrays, df, config.test_fraction)
        model, norm_mean, norm_std = train_one_target(
            train_seqs, test_seqs, spec, config
        )
        save_artifacts(feature_columns, norm_mean, norm_std, spec, config)
        trained[spec.suffix] = (model, norm_mean, norm_std)

    if config.predict_to is None:
        return

    update_status("generating predictions for full dataset")
    predictions = df.copy()
    for spec in target_specs:
        model, norm_mean, norm_std = trained[spec.suffix]
        all_arrays = build_sequences(
            df,
            feature_columns,
            spec.column,
            config.seq_len,
            training_only=False,
        )
        proba = run_predictions(all_arrays, model, norm_mean, norm_std, config.batch_size)
        predictions = add_bucket_predictions(predictions, all_arrays, proba, spec)

    save_parquet(predictions, config.predict_to)
    update_status(f"saved predictions to {config.predict_to}")


if __name__ == "__main__":
    main()
