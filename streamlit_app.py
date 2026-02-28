import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Hugging Face repo with your 3 pkl files
HF_REPO = "shiavm006/Crop-yield_pridiction"
MODEL_FILENAME = "model.pkl"
SCALER_FILENAME = "scaler.pkl"
FEATURES_FILENAME = "features.pkl"


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_artifacts_from_hf():
    """Load model, scaler, and features from Hugging Face Hub (or local model/ if present)."""
    base = Path(__file__).parent / "model"
    if (base / MODEL_FILENAME).exists():
        return (
            _load_pkl(base / MODEL_FILENAME),
            _load_pkl(base / SCALER_FILENAME),
            _load_pkl(base / FEATURES_FILENAME),
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        st.error("Install huggingface_hub: pip install huggingface_hub")
        return None, None, None

    try:
        model_path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILENAME)
        scaler_path = hf_hub_download(repo_id=HF_REPO, filename=SCALER_FILENAME)
        features_path = hf_hub_download(repo_id=HF_REPO, filename=FEATURES_FILENAME)
    except Exception as e:
        st.error(f"Could not download from Hugging Face: {e}")
        return None, None, None

    return _load_pkl(model_path), _load_pkl(scaler_path), _load_pkl(features_path)


@st.cache_data
def load_dropdown_options():
    """Load unique Area and Item from dataset for dropdowns."""
    csv_path = Path(__file__).parent / "Dataset" / "yield_df.csv"
    if not csv_path.exists():
        return [], []
    df = pd.read_csv(csv_path)
    areas = sorted(df["Area"].dropna().unique().tolist())
    items = sorted(df["Item"].dropna().unique().tolist())
    return areas, items


def get_feature_names(features_config):
    """Get list of feature names in model order from features.pkl."""
    if features_config is None:
        return None
    if isinstance(features_config, list):
        return features_config
    if isinstance(features_config, dict):
        return features_config.get("feature_names") or features_config.get("columns") or features_config.get("names")
    if hasattr(features_config, "get_feature_names_out"):
        return features_config.get_feature_names_out().tolist()
    return None


def build_input_row(area, item, year, rainfall, pesticides, avg_temp, feature_names):
    """
    Build a single row (array) in the exact order expected by the model.
    Handles: (1) numeric-only features, (2) numeric + one-hot Area/Item.
    """
    import numpy as np

    numeric = {
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": avg_temp,
    }
    values = []
    for name in feature_names:
        if name in numeric:
            values.append(numeric[name])
        elif name.startswith("Area_"):
            values.append(1.0 if name == f"Area_{area}" else 0.0)
        elif name.startswith("Item_"):
            # Match "Item_Maize" or "Item_Rice, paddy"
            match = name == f"Item_{item}" or item in name.replace("_", " ")
            values.append(1.0 if match else 0.0)
        else:
            values.append(0.0)
    return np.array([values], dtype=float)


def main():
    st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="centered")
    st.title("🌾 Crop Yield Prediction")
    st.caption("Milestone 1 — ML-based yield prediction from farm, soil & weather data")

    model, scaler, features_config = load_artifacts_from_hf()
    if model is None:
        st.stop()

    feature_names = get_feature_names(features_config)
    if feature_names is None:
        st.warning("Could not get feature names from features.pkl. Using default numeric-only order.")
        feature_names = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]

    areas, items = load_dropdown_options()
    if not areas:
        areas = ["Albania", "India", "United States of America"]
    if not items:
        items = ["Maize", "Potatoes", "Rice, paddy", "Sorghum", "Soybeans", "Wheat"]

    with st.form("yield_form"):
        st.subheader("Farm & weather inputs")
        col1, col2 = st.columns(2)
        with col1:
            area = st.selectbox("Area (country/region)", options=areas, index=min(0, len(areas) - 1))
            item = st.selectbox("Crop (Item)", options=items, index=min(0, len(items) - 1))
            year = st.number_input("Year", min_value=1960, max_value=2030, value=2020, step=1)
        with col2:
            rainfall = st.number_input("Rainfall (mm/year)", min_value=0.0, value=1000.0, step=50.0, format="%.1f")
            pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, value=100.0, step=10.0, format="%.1f")
            avg_temp = st.number_input("Average temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.5, format="%.2f")

        submitted = st.form_submit_button("Predict yield")

    if submitted:
        try:
            X = build_input_row(area, item, year, rainfall, pesticides, avg_temp, feature_names)
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            st.success(f"**Predicted yield:** {pred:,.0f} hg/ha")
            st.caption("Yield is in hectogram per hectare (hg/ha). 100 hg/ha = 1 tonne/ha.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            if st.checkbox("Show debug info"):
                st.code(str(e))
                st.json({"feature_names": feature_names})


if __name__ == "__main__":
    main()
