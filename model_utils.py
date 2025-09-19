# model_utils.py
import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

def load_class_map(path="model/class_map.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            m = json.load(f)
        # If mapping is a dict with numeric keys as strings -> convert to list
        if isinstance(m, dict):
            try:
                keys = sorted(m.keys(), key=lambda x: int(x))
                return [m[k] for k in keys]
            except Exception:
                # fallback to values order
                return list(m.values())
        elif isinstance(m, list):
            return m
        else:
            raise ValueError("class_map.json has unknown format")
    else:
        raise FileNotFoundError(f"class_map.json not found at {path}")

def load_rice_model(model_path_h5="model/rice_classification_model.h5",
                    saved_model_dir="model/rice_saved_model"):
    # Prefer .h5 if present else saved_model folder
    if os.path.exists(model_path_h5):
        model = load_model(model_path_h5)
        model_format = "h5"
    elif os.path.exists(saved_model_dir):
        model = load_model(saved_model_dir)
        model_format = "saved_model"
    else:
        raise FileNotFoundError("No model found. Place .h5 or SavedModel in the model/ folder.")
    return model, model_format

def get_model_input_size(model):
    # model.input_shape often like (None, H, W, 3)
    shape = getattr(model, "input_shape", None)
    if not shape:
        return (128, 128)
    # try typical form
    try:
        _, h, w, c = shape
        if h is None or w is None:
            return (128, 128)
        return (int(h), int(w))
    except Exception:
        # try other shape forms
        try:
            return (int(shape[-3]), int(shape[-2]))
        except Exception:
            return (128, 128)

def preprocess_pil_image(pil_img: Image.Image, target_size):
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return np.expand_dims(arr, axis=0)

def predict_topk(model, pil_img, class_names, k=3):
    target_size = get_model_input_size(model)
    x = preprocess_pil_image(pil_img, target_size)
    preds = model.predict(x)[0]
    topk_idx = preds.argsort()[-k:][::-1]
    results = [{"label": class_names[i], "confidence": float(preds[i])} for i in topk_idx]
    return results
