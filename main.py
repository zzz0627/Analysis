from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names")

TARGET_COLUMN = "subscribe"
ID_COLUMN = "id"
AVAILABLE_MODELS = ("catboost", "lightgbm", "xgboost")
MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
DOW_MAP = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5}
MACRO_COLUMNS = [
    "emp_var_rate",
    "cons_price_index",
    "cons_conf_index",
    "lending_rate3m",
    "nr_employed",
]


@dataclass(frozen=True)
class RunConfig:
    train_path: Path
    test_path: Path
    sample_path: Path
    output_root: Path
    n_splits: int
    seed: int
    use_duration: bool
    models: tuple[str, ...]
    weight_step: float


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Train an ensemble for bank product subscription prediction."
    )
    parser.add_argument("--train-path", default="dataset/train.csv")
    parser.add_argument("--test-path", default="dataset/test.csv")
    parser.add_argument("--sample-path", default="dataset/submission.csv")
    parser.add_argument("--output-root", default="submission")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        default="catboost,lightgbm,xgboost",
        help="Comma separated subset of: catboost,lightgbm,xgboost",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.1,
        help="Blend weight search step. Smaller is slower but more precise.",
    )
    parser.add_argument(
        "--drop-duration",
        action="store_true",
        help="Disable duration and duration-derived features for a leakage-safer run.",
    )
    args = parser.parse_args()

    models = tuple(
        dict.fromkeys(
            model.strip().lower() for model in args.models.split(",") if model.strip()
        )
    )
    if not models:
        raise ValueError("At least one model must be selected.")

    invalid_models = [model for model in models if model not in AVAILABLE_MODELS]
    if invalid_models:
        raise ValueError(
            f"Unsupported model(s): {invalid_models}. Available: {AVAILABLE_MODELS}"
        )

    if args.n_splits < 3:
        raise ValueError("--n-splits must be at least 3.")

    if not 0 < args.weight_step <= 1:
        raise ValueError("--weight-step must be in the interval (0, 1].")

    return RunConfig(
        train_path=Path(args.train_path),
        test_path=Path(args.test_path),
        sample_path=Path(args.sample_path),
        output_root=Path(args.output_root),
        n_splits=args.n_splits,
        seed=args.seed,
        use_duration=not args.drop_duration,
        models=models,
        weight_step=args.weight_step,
    )


def load_datasets(config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (config.train_path, config.test_path, config.sample_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    sample_df = pd.read_csv(config.sample_path)

    required_train = {
        ID_COLUMN,
        TARGET_COLUMN,
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        *MACRO_COLUMNS,
    }
    required_test = required_train - {TARGET_COLUMN}

    missing_train = sorted(required_train - set(train_df.columns))
    missing_test = sorted(required_test - set(test_df.columns))
    if missing_train:
        raise ValueError(f"Train columns missing: {missing_train}")
    if missing_test:
        raise ValueError(f"Test columns missing: {missing_test}")

    if train_df[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"Expected binary labels in `{TARGET_COLUMN}`.")

    return train_df, test_df, sample_df


def build_features(df: pd.DataFrame, use_duration: bool) -> pd.DataFrame:
    features = df.copy()

    features["month_num"] = features["month"].map(MONTH_MAP).fillna(0).astype(int)
    features["dow_num"] = features["day_of_week"].map(DOW_MAP).fillna(0).astype(int)

    month_angle = 2.0 * np.pi * (features["month_num"] - 1) / 12.0
    dow_angle = 2.0 * np.pi * (features["dow_num"] - 1) / 5.0
    features["month_sin"] = np.sin(month_angle)
    features["month_cos"] = np.cos(month_angle)
    features["dow_sin"] = np.sin(dow_angle)
    features["dow_cos"] = np.cos(dow_angle)

    features["is_contacted_before"] = (features["pdays"] < 999).astype(int)
    features["never_contacted_before"] = (features["pdays"] >= 999).astype(int)
    features["has_previous_contacts"] = (features["previous"] > 0).astype(int)
    features["pdays_clean"] = np.where(features["pdays"] >= 999, -1, features["pdays"])
    features["pdays_bucket"] = pd.cut(
        features["pdays_clean"],
        bins=[-2, -1, 7, 30, 90, 365, 5000],
        labels=["never", "1w", "1m", "3m", "1y", "old"],
        include_lowest=True,
    ).astype(str)

    features["campaign_log"] = np.log1p(features["campaign"].clip(lower=0))
    features["previous_log"] = np.log1p(features["previous"].clip(lower=0))
    features["contacts_total"] = features["campaign"] + features["previous"]
    features["contacts_total_log"] = np.log1p(features["contacts_total"].clip(lower=0))
    features["previous_share"] = features["previous"] / (
        features["contacts_total"] + 1.0
    )

    features["age_bucket"] = pd.cut(
        features["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
        include_lowest=True,
    ).astype(str)

    macro_frame = features[MACRO_COLUMNS]
    features["macro_mean"] = macro_frame.mean(axis=1)
    features["macro_std"] = macro_frame.std(axis=1)
    features["macro_min"] = macro_frame.min(axis=1)
    features["macro_max"] = macro_frame.max(axis=1)
    features["macro_range"] = features["macro_max"] - features["macro_min"]
    features["rate_x_employment"] = (
        features["emp_var_rate"] * features["nr_employed"]
    )
    features["price_x_conf"] = (
        features["cons_price_index"] * features["cons_conf_index"]
    )
    features["price_minus_rate"] = (
        features["cons_price_index"] - features["lending_rate3m"]
    )

    features["job_marital"] = (
        features["job"].astype(str) + "__" + features["marital"].astype(str)
    )
    features["education_default"] = (
        features["education"].astype(str) + "__" + features["default"].astype(str)
    )
    features["housing_loan"] = (
        features["housing"].astype(str) + "__" + features["loan"].astype(str)
    )
    features["contact_month"] = (
        features["contact"].astype(str) + "__" + features["month"].astype(str)
    )
    features["poutcome_contacted"] = (
        features["poutcome"].astype(str)
        + "__"
        + features["is_contacted_before"].astype(str)
    )

    if use_duration:
        features["duration_log"] = np.log1p(features["duration"].clip(lower=0))
        features["contact_intensity"] = features["duration"] / (
            features["campaign"] + 1.0
        )
        features["contact_intensity_log"] = np.log1p(
            features["contact_intensity"].clip(lower=0)
        )
        features["duration_per_previous"] = features["duration"] / (
            features["previous"] + 1.0
        )
        features["duration_per_previous_log"] = np.log1p(
            features["duration_per_previous"].clip(lower=0)
        )
    else:
        features = features.drop(columns=["duration"])

    object_columns = features.select_dtypes(include=["object"]).columns
    for column in object_columns:
        features[column] = features[column].fillna("missing").astype(str)

    return features


def prepare_training_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_duration: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], list[str]]:
    train_features = build_features(train_df, use_duration=use_duration)
    test_features = build_features(test_df, use_duration=use_duration)

    y = train_features[TARGET_COLUMN].map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError(
            f"Unexpected labels in `{TARGET_COLUMN}`: "
            f"{sorted(train_features[TARGET_COLUMN].unique().tolist())}"
        )

    test_ids = test_features[ID_COLUMN].copy()
    train_matrix = train_features.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    test_matrix = test_features.drop(columns=[ID_COLUMN])
    test_matrix = test_matrix[train_matrix.columns]

    categorical_columns = train_matrix.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = [
        column for column in train_matrix.columns if column not in categorical_columns
    ]
    return train_matrix, y.astype(int), test_matrix, test_ids, categorical_columns, numeric_columns


def build_preprocessor(
    categorical_columns: list[str], numeric_columns: list[str]
) -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_model(
    model_name: str,
    categorical_columns: list[str],
    numeric_columns: list[str],
    seed: int,
):
    if model_name == "catboost":
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Accuracy",
            iterations=700,
            learning_rate=0.045,
            depth=6,
            l2_leaf_reg=8,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            random_strength=0.8,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

    preprocessor = build_preprocessor(categorical_columns, numeric_columns)
    if model_name == "lightgbm":
        estimator = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            objective="binary",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        return Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", estimator)]
        )

    if model_name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=450,
            learning_rate=0.04,
            max_depth=5,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            tree_method="hist",
            max_bin=256,
            n_jobs=-1,
        )
        return Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", estimator)]
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def build_threshold_grid() -> np.ndarray:
    return np.round(np.arange(0.05, 0.951, 0.01), 2)


def search_best_threshold(
    y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> tuple[float, float]:
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_true, predictions)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold, float(best_accuracy)


def generate_weight_combinations(
    model_names: list[str], step: float
) -> list[dict[str, float]]:
    if len(model_names) == 1:
        return [{model_names[0]: 1.0}]

    total_units = int(round(1.0 / step))
    combinations: list[dict[str, float]] = []

    def backtrack(index: int, units_left: int, current: dict[str, float]) -> None:
        if index == len(model_names) - 1:
            current[model_names[index]] = units_left / total_units
            combinations.append(current.copy())
            return

        for unit in range(units_left + 1):
            current[model_names[index]] = unit / total_units
            backtrack(index + 1, units_left - unit, current)

    backtrack(0, total_units, {})
    return combinations


def search_best_ensemble(
    y_true: np.ndarray,
    model_oof_predictions: dict[str, np.ndarray],
    thresholds: np.ndarray,
    weight_step: float,
) -> tuple[dict[str, float], float, float]:
    model_names = list(model_oof_predictions.keys())
    weight_candidates = generate_weight_combinations(model_names, step=weight_step)

    best_weights = {model_names[0]: 1.0}
    best_threshold = 0.5
    best_accuracy = -1.0

    for weights in weight_candidates:
        blended_predictions = np.zeros_like(next(iter(model_oof_predictions.values())))
        for model_name, weight in weights.items():
            blended_predictions += weight * model_oof_predictions[model_name]

        threshold, accuracy = search_best_threshold(y_true, blended_predictions, thresholds)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_weights = weights.copy()

    return best_weights, float(best_threshold), float(best_accuracy)


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    config: RunConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    splitter = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    thresholds = build_threshold_grid()

    oof_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, Any]] = {}

    for model_name in config.models:
        print(f"\n[Model] {model_name}")
        model_oof = np.zeros(len(X), dtype=float)
        fold_test_predictions = np.zeros((config.n_splits, len(X_test)), dtype=float)
        fold_scores: list[float] = []

        for fold_index, (train_idx, valid_idx) in enumerate(
            splitter.split(X, y), start=1
        ):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_valid = y.iloc[valid_idx]

            model = build_model(
                model_name,
                categorical_columns=categorical_columns,
                numeric_columns=numeric_columns,
                seed=config.seed + fold_index,
            )

            if model_name == "catboost":
                model.fit(X_train, y_train, cat_features=categorical_columns)
            else:
                model.fit(X_train, y_train)

            valid_probabilities = model.predict_proba(X_valid)[:, 1]
            test_probabilities = model.predict_proba(X_test)[:, 1]

            model_oof[valid_idx] = valid_probabilities
            fold_test_predictions[fold_index - 1] = test_probabilities

            fold_threshold, fold_accuracy = search_best_threshold(
                y_valid.to_numpy(), valid_probabilities, thresholds
            )
            fold_scores.append(fold_accuracy)
            print(
                f"  Fold {fold_index}: accuracy={fold_accuracy:.5f} "
                f"best_threshold={fold_threshold:.2f}"
            )

        best_threshold, best_accuracy = search_best_threshold(
            y.to_numpy(), model_oof, thresholds
        )
        oof_predictions[model_name] = model_oof
        test_predictions[model_name] = fold_test_predictions.mean(axis=0)
        metrics[model_name] = {
            "oof_accuracy": float(best_accuracy),
            "best_threshold": float(best_threshold),
            "mean_fold_accuracy": float(np.mean(fold_scores)),
            "std_fold_accuracy": float(np.std(fold_scores)),
        }
        print(
            f"[OOF] {model_name}: accuracy={best_accuracy:.5f}, "
            f"threshold={best_threshold:.2f}"
        )

    return oof_predictions, test_predictions, metrics


def build_submission(
    sample_df: pd.DataFrame, test_ids: pd.Series, predictions: np.ndarray
) -> pd.DataFrame:
    if ID_COLUMN in sample_df.columns:
        id_column = ID_COLUMN
    else:
        id_column = sample_df.columns[0]

    target_columns = [column for column in sample_df.columns if column != id_column]
    target_column = target_columns[0] if target_columns else TARGET_COLUMN

    prediction_df = pd.DataFrame(
        {id_column: test_ids.to_numpy(), target_column: predictions}
    )

    if id_column in sample_df.columns and len(sample_df) == len(prediction_df):
        merged = sample_df[[id_column]].merge(prediction_df, on=id_column, how="left")
        if merged[target_column].isna().any():
            raise ValueError("Submission template IDs do not match test IDs.")
        return merged[[id_column, target_column]]

    return prediction_df[[id_column, target_column]]


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def save_outputs(
    output_dir: Path,
    submission_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_dir / "submission.csv", index=False)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(to_serializable(summary), file, indent=2, ensure_ascii=False)


def main() -> None:
    config = parse_args()
    np.random.seed(config.seed)

    print("Loading data...")
    train_df, test_df, sample_df = load_datasets(config)

    X, y, X_test, test_ids, categorical_columns, numeric_columns = (
        prepare_training_matrices(
            train_df=train_df,
            test_df=test_df,
            use_duration=config.use_duration,
        )
    )

    positive_rate = float(y.mean())
    print(
        f"Train shape={train_df.shape}, Test shape={test_df.shape}, "
        f"Positive rate={positive_rate:.4f}"
    )
    print(
        f"Features={X.shape[1]}, categorical={len(categorical_columns)}, "
        f"numeric={len(numeric_columns)}, use_duration={config.use_duration}"
    )
    print(f"Models={list(config.models)}")

    oof_predictions, test_predictions, model_metrics = run_cross_validation(
        X=X,
        y=y,
        X_test=X_test,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        config=config,
    )

    thresholds = build_threshold_grid()
    best_weights, best_threshold, best_accuracy = search_best_ensemble(
        y_true=y.to_numpy(),
        model_oof_predictions=oof_predictions,
        thresholds=thresholds,
        weight_step=config.weight_step,
    )

    blended_test_probabilities = np.zeros(len(X_test), dtype=float)
    for model_name, weight in best_weights.items():
        blended_test_probabilities += weight * test_predictions[model_name]

    final_predictions = np.where(blended_test_probabilities >= best_threshold, "yes", "no")
    submission_df = build_submission(
        sample_df=sample_df,
        test_ids=test_ids,
        predictions=final_predictions,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_root / f"submission_{timestamp}"

    summary = {
        "timestamp": timestamp,
        "config": {
            "train_path": config.train_path,
            "test_path": config.test_path,
            "sample_path": config.sample_path,
            "output_root": config.output_root,
            "n_splits": config.n_splits,
            "seed": config.seed,
            "use_duration": config.use_duration,
            "models": config.models,
            "weight_step": config.weight_step,
        },
        "dataset": {
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "positive_rate": positive_rate,
            "feature_count": int(X.shape[1]),
            "categorical_count": len(categorical_columns),
            "numeric_count": len(numeric_columns),
        },
        "model_metrics": model_metrics,
        "ensemble": {
            "oof_accuracy": float(best_accuracy),
            "best_threshold": float(best_threshold),
            "weights": {
                model_name: float(weight)
                for model_name, weight in best_weights.items()
                if weight > 0
            },
        },
        "artifacts": {
            "submission_path": output_dir / "submission.csv",
            "summary_path": output_dir / "run_summary.json",
        },
    }
    save_outputs(output_dir=output_dir, submission_df=submission_df, summary=summary)

    print("\n[Ensemble]")
    print(
        f"OOF accuracy={best_accuracy:.5f}, best_threshold={best_threshold:.2f}, "
        f"weights={summary['ensemble']['weights']}"
    )
    print(f"Submission saved to: {output_dir / 'submission.csv'}")
    print(f"Run summary saved to: {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
