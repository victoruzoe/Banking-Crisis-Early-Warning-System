from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PUBLIC_DATA_URL = (
    "https://raw.githubusercontent.com/RAFrancais/"
    "Africa-economic-data-kaggle/master/african_crises.csv"
)

TARGET_COLUMN = "banking_crisis_next_year"
YEAR_COLUMN = "target_year"

RAW_FEATURE_COLUMNS = [
    "country",
    "systemic_crisis",
    "exch_usd",
    "domestic_debt_in_default",
    "sovereign_external_debt_default",
    "gdp_weighted_default",
    "inflation_annual_cpi",
    "independence",
    "currency_crises",
    "inflation_crises",
]

MODEL_FEATURE_COLUMNS = [
    "country",
    "systemic_crisis",
    "exch_usd_log1p",
    "domestic_debt_in_default",
    "sovereign_external_debt_default",
    "gdp_weighted_default",
    "inflation_signed_log1p",
    "independence",
    "currency_crises_code",
    "inflation_crises",
]

CATEGORICAL_COLUMNS = ["country", "currency_crises_code"]
CONTINUOUS_COLUMNS = [
    "exch_usd_log1p",
    "gdp_weighted_default",
    "inflation_signed_log1p",
]
BINARY_COLUMNS = [
    "systemic_crisis",
    "domestic_debt_in_default",
    "sovereign_external_debt_default",
    "independence",
    "inflation_crises",
]


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root from the working directory or this module's location."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists() and (candidate / "src").exists():
            return candidate

    module_root = Path(__file__).resolve().parents[1]
    if (module_root / "README.md").exists():
        return module_root
    return current


def load_raw_data(local_path: str | Path | None = None) -> pd.DataFrame:
    """Load the source dataset from a local CSV when available, otherwise from a public mirror."""
    candidates: List[Path] = []
    if local_path is not None:
        candidates.append(Path(local_path))

    root = find_project_root()
    candidates.extend(
        [
            root / "data" / "African_crises_dataset.csv",
            root / "data" / "african_crises.csv",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return standardize_schema(pd.read_csv(candidate))

    try:
        return standardize_schema(pd.read_csv(PUBLIC_DATA_URL))
    except Exception as exc:
        raise RuntimeError(
            "The African Crises dataset could not be loaded. Place the CSV in data/ as "
            "African_crises_dataset.csv or african_crises.csv, then try again."
        ) from exc


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column-name variants used by copies of the dataset."""
    rename_map = {}
    if "case" in df.columns and "country_number" not in df.columns:
        rename_map["case"] = "country_number"
    if "cc3" in df.columns and "country_code" not in df.columns:
        rename_map["cc3"] = "country_code"
    clean = df.rename(columns=rename_map).copy()

    required = {
        "country_number",
        "country_code",
        "country",
        "year",
        "systemic_crisis",
        "exch_usd",
        "domestic_debt_in_default",
        "sovereign_external_debt_default",
        "gdp_weighted_default",
        "inflation_annual_cpi",
        "independence",
        "currency_crises",
        "inflation_crises",
        "banking_crisis",
    }
    missing = sorted(required.difference(clean.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return clean


def clean_source_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply transparent source-data checks without using future target information."""
    clean = standardize_schema(df).copy()
    clean = clean.drop_duplicates().copy()

    clean["banking_crisis"] = clean["banking_crisis"].astype(str).str.strip().str.lower()
    valid_target = {"crisis", "no_crisis"}
    clean = clean[clean["banking_crisis"].isin(valid_target)].copy()

    # The source documentation defines currency_crises as binary. Four rows use code 2.
    clean = clean[clean["currency_crises"].isin([0, 1])].copy()

    clean = clean.sort_values(["country", "year"]).reset_index(drop=True)
    return clean


def prepare_early_warning_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-year-ahead early-warning dataset.

    Features come from country-year t and the target is banking-crisis status at t+1.
    Rows are kept only when the next observation for that country is exactly one year later.
    """
    clean = clean_source_data(df)
    clean["banking_crisis_binary"] = clean["banking_crisis"].map(
        {"no_crisis": 0, "crisis": 1}
    )

    grouped = clean.groupby("country", sort=False)
    clean["next_year"] = grouped["year"].shift(-1)
    clean[TARGET_COLUMN] = grouped["banking_crisis_binary"].shift(-1)

    consecutive = clean["next_year"].eq(clean["year"] + 1)
    frame = clean.loc[consecutive].copy()
    frame[YEAR_COLUMN] = frame["next_year"].astype(int)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)

    return frame.reset_index(drop=True)


def build_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, row-wise transformations used by both analysis and inference."""
    features = frame[RAW_FEATURE_COLUMNS].copy()

    exch = pd.to_numeric(features["exch_usd"], errors="coerce").clip(lower=0)
    inflation = pd.to_numeric(features["inflation_annual_cpi"], errors="coerce")

    features["exch_usd_log1p"] = np.log1p(exch)
    features["inflation_signed_log1p"] = np.sign(inflation) * np.log1p(np.abs(inflation))
    features["currency_crises_code"] = (
        pd.to_numeric(features["currency_crises"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )

    return features[MODEL_FEATURE_COLUMNS]


def make_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    continuous = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, CATEGORICAL_COLUMNS),
            ("continuous", continuous, CONTINUOUS_COLUMNS),
            ("binary", binary, BINARY_COLUMNS),
        ],
        remainder="drop",
    )


def candidate_estimators(include_dummy: bool = True) -> Dict[str, object]:
    models: Dict[str, object] = {}
    if include_dummy:
        models["Dummy Prior"] = DummyClassifier(strategy="prior")
    models.update(
        {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=42,
            ),
            "Decision Tree": DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=5,
                min_samples_leaf=5,
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                max_depth=7,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
            "SVC": SVC(
                C=1.0,
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
        }
    )
    return models


def make_model_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ]
    )


def temporal_holdout_split(
    frame: pd.DataFrame,
    test_fraction: float = 0.20,
    min_positive: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split by target year, reserving the latest years as a genuine historical holdout."""
    data = frame.sort_values([YEAR_COLUMN, "country"]).reset_index(drop=True)
    years = np.array(sorted(data[YEAR_COLUMN].unique()))
    if len(years) < 8:
        raise ValueError("Not enough unique years for a temporal holdout.")

    preferred = int(round((1 - test_fraction) * len(years)))
    candidate_positions = sorted(
        range(max(2, int(0.65 * len(years))), min(len(years) - 2, int(0.90 * len(years))) + 1),
        key=lambda i: abs(i - preferred),
    )

    for position in candidate_positions:
        cutoff = int(years[position])
        train = data[data[YEAR_COLUMN] < cutoff].copy()
        test = data[data[YEAR_COLUMN] >= cutoff].copy()
        if train.empty or test.empty:
            continue
        if train[TARGET_COLUMN].nunique() < 2 or test[TARGET_COLUMN].nunique() < 2:
            continue
        if int(test[TARGET_COLUMN].sum()) < min_positive:
            continue
        return train, test, cutoff

    raise ValueError("Could not create a temporal holdout containing both target classes.")


def expanding_year_splits(
    train_frame: pd.DataFrame,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window validation folds using complete target-year blocks."""
    data = train_frame.reset_index(drop=True)
    years = np.array(sorted(data[YEAR_COLUMN].unique()))
    blocks = [block for block in np.array_split(years, n_splits + 1) if len(block)]

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, len(blocks)):
        train_years = np.concatenate(blocks[:i])
        valid_years = blocks[i]
        tr_idx = np.flatnonzero(data[YEAR_COLUMN].isin(train_years).to_numpy())
        va_idx = np.flatnonzero(data[YEAR_COLUMN].isin(valid_years).to_numpy())
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        y_tr = data.iloc[tr_idx][TARGET_COLUMN]
        y_va = data.iloc[va_idx][TARGET_COLUMN]
        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            continue
        splits.append((tr_idx, va_idx))

    if len(splits) < 2:
        raise ValueError("Not enough valid expanding-year folds containing both classes.")
    return splits


def _positive_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    scores = np.asarray(scores, dtype=float)
    return 1.0 / (1.0 + np.exp(-scores))


def compare_models(
    train_frame: pd.DataFrame,
    n_splits: int = 5,
    include_dummy: bool = True,
) -> pd.DataFrame:
    """Compare candidate classifiers using expanding-year cross-validation."""
    ordered = train_frame.sort_values([YEAR_COLUMN, "country"]).reset_index(drop=True)
    X_all = build_model_features(ordered)
    y_all = ordered[TARGET_COLUMN].astype(int)
    splits = expanding_year_splits(ordered, n_splits=n_splits)

    rows: List[dict] = []
    for model_name, estimator in candidate_estimators(include_dummy=include_dummy).items():
        fold_metrics: List[dict] = []
        for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
            pipeline = make_model_pipeline(clone(estimator))
            X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
            y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]
            pipeline.fit(X_tr, y_tr)
            pred = pipeline.predict(X_va)
            score = _positive_scores(pipeline, X_va)
            fold_metrics.append(
                {
                    "fold": fold,
                    "roc_auc": roc_auc_score(y_va, score),
                    "pr_auc": average_precision_score(y_va, score),
                    "accuracy": accuracy_score(y_va, pred),
                    "balanced_accuracy": balanced_accuracy_score(y_va, pred),
                    "precision": precision_score(y_va, pred, zero_division=0),
                    "recall": recall_score(y_va, pred, zero_division=0),
                    "f1": f1_score(y_va, pred, zero_division=0),
                }
            )

        metric_frame = pd.DataFrame(fold_metrics)
        row = {"model": model_name, "folds": len(metric_frame)}
        for metric in [
            "roc_auc",
            "pr_auc",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
        ]:
            row[f"mean_{metric}"] = metric_frame[metric].mean()
            row[f"std_{metric}"] = metric_frame[metric].std(ddof=0)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)


def select_model_name(comparison: pd.DataFrame, simplicity_margin: float = 0.02) -> str:
    """Prefer Logistic Regression when its ROC-AUC is within a small margin of the best model."""
    candidates = comparison[comparison["model"] != "Dummy Prior"].copy()
    if candidates.empty:
        raise ValueError("No non-dummy models are available for selection.")

    best_row = candidates.sort_values("mean_roc_auc", ascending=False).iloc[0]
    if "Logistic Regression" in candidates["model"].values:
        logistic = candidates[candidates["model"] == "Logistic Regression"].iloc[0]
        if float(best_row["mean_roc_auc"] - logistic["mean_roc_auc"]) <= simplicity_margin:
            return "Logistic Regression"
    return str(best_row["model"])


def fit_selected_model(train_frame: pd.DataFrame, model_name: str) -> Pipeline:
    estimators = candidate_estimators(include_dummy=False)
    if model_name not in estimators:
        raise KeyError(f"Unknown model name: {model_name}")
    pipeline = make_model_pipeline(clone(estimators[model_name]))
    pipeline.fit(build_model_features(train_frame), train_frame[TARGET_COLUMN].astype(int))
    return pipeline


def evaluate_holdout(model: Pipeline, test_frame: pd.DataFrame) -> dict:
    X_test = build_model_features(test_frame)
    y_test = test_frame[TARGET_COLUMN].astype(int)
    pred = model.predict(X_test)
    score = _positive_scores(model, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, score),
        "pr_auc": average_precision_score(y_test, score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fit_deployment_model(
    frame: pd.DataFrame,
    n_splits: int = 5,
) -> Tuple[Pipeline, dict]:
    """
    Select a model using historical training data, evaluate once on a later holdout,
    then refit the selected specification on all labelled observations for demo inference.
    """
    train, test, cutoff = temporal_holdout_split(frame)
    comparison = compare_models(train, n_splits=n_splits, include_dummy=True)
    selected_name = select_model_name(comparison)

    evaluation_model = fit_selected_model(train, selected_name)
    holdout_metrics = evaluate_holdout(evaluation_model, test)

    deployment_model = fit_selected_model(frame, selected_name)
    metadata = {
        "selected_model": selected_name,
        "holdout_cutoff_year": int(cutoff),
        "holdout_metrics": holdout_metrics,
        "comparison": comparison.to_dict(orient="records"),
        "n_observations": int(len(frame)),
        "n_crisis_targets": int(frame[TARGET_COLUMN].sum()),
        "target_rate": float(frame[TARGET_COLUMN].mean()),
    }
    return deployment_model, metadata


def feature_effects(model: Pipeline) -> pd.DataFrame:
    """Return coefficients or feature importances for the fitted final estimator when available."""
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)

    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_).reshape(-1)
        kind = "coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).reshape(-1)
        kind = "importance"
    else:
        return pd.DataFrame(columns=["feature", "value", "kind", "abs_value"])

    out = pd.DataFrame({"feature": names, "value": values})
    out["kind"] = kind
    out["abs_value"] = out["value"].abs()
    return out.sort_values("abs_value", ascending=False).reset_index(drop=True)
