from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

# Đọc tên bucket từ biến môi trường
GCS_BUCKET = os.environ.get("GCS_BUCKET")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """Tải file model.pkl từ GCS về máy khi server khởi động."""
    if not GCS_BUCKET:
        print("GCS_BUCKET environment variable not set. Skipping model download.")
        return

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print(f"Model downloaded successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"Error downloading model: {e}")


# Gọi hàm này khi server khởi động
if __name__ == "__main__" or os.environ.get("KUBERNETES_SERVICE_HOST"): # Simple check if running in server context
    download_model()

# Load model if it exists
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe server."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    prediction = int(model.predict([req.features])[0])
    labels = {0: "thấp", 1: "trung_bình", 2: "cao"}
    
    return {
        "prediction": prediction,
        "label": labels.get(prediction, "unknown")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
