import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop("target", axis=1)
    y_train = df_train["target"]
    X_eval = df_eval.drop("target", axis=1)
    y_eval = df_eval["target"]

    model_type = params.get("model_type", "random_forest")
    model_params = {k: v for k, v in params.items() if k != "model_type"}

    with mlflow.start_run():
        mlflow.log_params(params)

        if model_type == "random_forest":
            model = RandomForestClassifier(**model_params, random_state=42)
        elif model_type == "extra_trees":
            model = ExtraTreesClassifier(**model_params, random_state=42)
        elif model_type == "hist_gradient_boosting":
            model = HistGradientBoostingClassifier(**model_params, random_state=42)
        else:
            model = RandomForestClassifier(**model_params, random_state=42)

        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"Model: {model_type} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    train(params)
