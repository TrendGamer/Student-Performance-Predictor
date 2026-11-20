"""Train and compare baseline regression models; save the winner and metrics."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from preprocessing import FEATURES, TARGET, CATEGORICAL_FEATURES, load_clean_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student_data.csv"
MODEL = ROOT / "models" / "student_performance_model.pkl"
METRICS = ROOT / "models" / "evaluation_metrics.json"


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(numeric_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", num_pipe, numeric),
        ("categorical", cat_pipe, CATEGORICAL_FEATURES),
    ])


def main() -> None:
    df = load_clean_data(DATA)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("regressor", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("regressor", RandomForestRegressor(
                n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1
            )),
        ]),
    }
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        scores[name] = {
            "MAE": float(mean_absolute_error(y_test, pred)),
            "MSE": float(mse),
            "RMSE": float(mse ** 0.5),
            "R2": float(r2_score(y_test, pred)),
        }
    winner = min(scores, key=lambda name: (scores[name]["MAE"], scores[name]["RMSE"]))
    best = models[winner]
    best.fit(X, y)  # deployable estimator uses all available cleaned rows
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "features": FEATURES, "winner": winner,
                 "synthetic_data": True}, MODEL)
    payload = {"dataset": "Reproducible synthetic demonstration data",
               "rows_after_duplicate_and_target_cleanup": len(df),
               "train_rows": len(X_train), "test_rows": len(X_test),
               "selection_rule": "Lowest holdout MAE; RMSE breaks ties",
               "selected_model": winner, "models": scores}
    METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved model: {MODEL}")


if __name__ == "__main__":
    main()
