from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI(title="ML Ops FastAPI Service")

# -------------------------------
# Cấu hình model và S3
# -------------------------------
S3_BUCKET = os.environ.get("GCS_BUCKET") or os.environ.get("S3_BUCKET")
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

# -------------------------------
# Download model từ S3
# -------------------------------
def download_model():
    """Tải model.pkl từ S3 về local khi server start"""
    if not S3_BUCKET:
        print("Bucket environment variable not set. Skipping model download.")
        return

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    try:
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
        print(f"[INFO] Model downloaded successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to download model from S3: {e}")

# Tải model ngay khi start server
download_model()

# -------------------------------
# Load model nếu tồn tại
# -------------------------------
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded from {MODEL_PATH}")
else:
    model = None
    print("[WARN] No model found. /predict sẽ trả 503")

# -------------------------------
# Request schema
# -------------------------------
class PredictRequest(BaseModel):
    features: list[float]

# -------------------------------
# Health check
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    prediction = int(model.predict([req.features])[0])
    labels = {0: "thấp", 1: "trung_bình", 2: "cao"}

    return {"prediction": prediction, "label": labels.get(prediction, "unknown")}

# -------------------------------
# Run server (for local testing)
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)