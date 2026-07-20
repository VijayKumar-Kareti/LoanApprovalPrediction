import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "loan_data.csv"
MODEL_PATH = PROJECT_ROOT / "model.pkl"

EXPECTED_FEATURES = [
    "Age",
    "Gender",
    "MaritalStatus",
    "Education",
    "EmploymentType",
    "MonthlyIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "LoanTerm",
    "CreditHistory",
    "PropertyArea",
    "Dependents",
    "SelfEmployed",
]

FEATURE_ALIASES = {
    "Age": ["age"],
    "Gender": ["gender"],
    "MaritalStatus": [
        "married",
        "marital_status",
        "marital status",
        "maritalstatus",
    ],
    "Education": ["education"],
    "EmploymentType": ["employment_type", "employmenttype"],
    "MonthlyIncome": [
        "applicantincome",
        "applicant_income",
        "monthly_income",
        "monthlyincome",
    ],
    "CoapplicantIncome": ["coapplicantincome", "coapplicant_income"],
    "LoanAmount": ["loanamount", "loan_amount"],
    "LoanTerm": ["loan_amount_term", "loanterm"],
    "CreditHistory": ["credit_history", "credithistory"],
    "PropertyArea": ["property_area", "propertyarea"],
    "Dependents": ["dependents"],
    "SelfEmployed": ["self_employed", "self employed", "selfemployed"],
}

TARGET_ALIASES = ["loan_status", "loanstatus", "loan status", "status"]


def normalize_column_name(col):
    return re.sub(r"[\s\-]+", "_", str(col).strip().lower())


def resolve_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized_columns = {normalize_column_name(col): col for col in df.columns}

    for alias in aliases:
        alias_key = normalize_column_name(alias)
        if alias_key in normalized_columns:
            return normalized_columns[alias_key]

    return None


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]

    # Rename common feature aliases to expected names
    for canonical_name, aliases in FEATURE_ALIASES.items():
        source_col = resolve_column(df, aliases)
        if source_col:
            df = df.rename(columns={source_col: canonical_name})

    # Rename target column to canonical Loan_Status
    target_col = resolve_column(df, TARGET_ALIASES)
    if target_col:
        df = df.rename(columns={target_col: "Loan_Status"})

    # Add missing columns if absent
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    if "Loan_Status" not in df.columns:
        raise KeyError(
            f"Could not find a loan status column. Available columns: {list(df.columns)}"
        )

    return df


def map_target(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()

    value_map = {
        "approved": 1,
        "y": 1,
        "yes": 1,
        "1": 1,
        "rejected": 0,
        "n": 0,
        "no": 0,
        "0": 0,
    }

    mapped = normalized.map(value_map)

    if mapped.isna().any():
        raise ValueError(
            f"Target contains unsupported values: {sorted(series.dropna().unique().tolist())}"
        )

    return mapped.astype(int)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_frame = df[EXPECTED_FEATURES].copy()

    numeric_cols = [
        "Age",
        "MonthlyIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "LoanTerm",
        "CreditHistory",
        "Dependents",
    ]

    for col in numeric_cols:
        feature_frame[col] = pd.to_numeric(feature_frame[col], errors="coerce")

    for col in numeric_cols:
        feature_frame[col] = feature_frame[col].fillna(0)

    categorical_cols = [col for col in EXPECTED_FEATURES if col not in numeric_cols]

    for col in categorical_cols:
        feature_frame[col] = feature_frame[col].fillna("Unknown").astype(str)

    X = feature_frame
    y = map_target(df["Loan_Status"])

    return X, y


def build_pipeline(model):
    numeric_features = [
        "Age",
        "MonthlyIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "LoanTerm",
        "CreditHistory",
        "Dependents",
    ]

    categorical_features = [
        col for col in EXPECTED_FEATURES if col not in numeric_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    trained_models = {}
    results = []

    for name, estimator in models.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        try:
            roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        except Exception:
            roc_auc = float("nan")

        confusion = confusion_matrix(y_test, y_pred)

        trained_models[name] = pipeline

        results.append(
            {
                "name": name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "confusion_matrix": confusion,
            }
        )

    return trained_models, results


def save_best_model(trained_models, results):
    best_result = max(results, key=lambda r: r["accuracy"])
    best_name = best_result["name"]
    best_pipeline = trained_models[best_name]

    bundle = {
        "model": best_pipeline,
        "feature_columns": EXPECTED_FEATURES,
    }

    joblib.dump(bundle, MODEL_PATH)

    print(f"Best model saved: {best_name}")
    print("Metrics:")
    for item in results:
        print(
            f"- {item['name']}: accuracy={item['accuracy']:.4f}, "
            f"precision={item['precision']:.4f}, recall={item['recall']:.4f}, "
            f"f1={item['f1']:.4f}, roc_auc={item['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    df = load_data(DATASET_PATH)
    X, y = prepare_features(df)

    trained_models, results = train_and_evaluate(X, y)
    save_best_model(trained_models, results)