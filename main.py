# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from model_utils import load_rice_model, load_class_map, predict_topk

app = FastAPI(title="Rice Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model once at startup
model, _ = load_rice_model()
class_names = load_class_map()

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")
    results = predict_topk(model, pil_img, class_names, k=top_k)
    return {"predictions": results}
