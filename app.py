# app.py
import streamlit as st
from PIL import Image
from model_utils import load_rice_model, load_class_map, predict_topk, get_model_input_size, preprocess_pil_image
import os

print("Current working directory:", os.getcwd())
print("Files in model/:", os.listdir("model"))

st.set_page_config(page_title="Rice Classifier", layout="centered")
st.title("🌾 Rice Type Classifier")
st.write("Upload an image of a rice grain — the model will predict its type.")

@st.cache_resource
def load_resources():
    model, model_format = load_rice_model()
    class_names = load_class_map()
    input_size = get_model_input_size(model)
    return model, class_names, input_size

try:
    model, class_names, input_size = load_resources()
except Exception as e:
    st.error(f"Model or class map not found. Place model files under `model/` folder. Error: {e}")
    st.stop()

st.sidebar.markdown("**Model info**")
st.sidebar.write(f"Input size: {input_size}")
st.sidebar.write(f"Classes: {len(class_names)}")

uploaded_file = st.file_uploader("Upload a rice image (jpg/png)", type=["jpg", "jpeg", "png"])
top_k = st.slider("Top K predictions", min_value=1, max_value=min(5, len(class_names)), value=3)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    # --- New code to show resized image ---
    resized_img = preprocess_pil_image(img, input_size)  # preprocess_pil_image should return a list/array
    st.image(resized_img[0], caption="Resized image for model", use_container_width=True)

    st.write(f"Model expects {input_size} (HxW). Uploaded image size: {img.size} (W x H).")

    if st.button("Classify"):
        with st.spinner("Predicting..."):
            results = predict_topk(model, img, class_names, k=top_k)
        st.success("Done")
        for r in results:
            st.write(f"**{r['label']}** — confidence: {r['confidence']:.3f}")
