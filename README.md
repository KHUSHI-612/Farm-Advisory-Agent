# Farm-Advisory-Agent

**Milestone 1:** ML-based Crop Yield Prediction (Project 8 — Intelligent Crop Yield Prediction & Agentic Farm Advisory)

## Run the app locally

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start Streamlit**
   ```bash
   streamlit run app.py
   ```
3. Open the URL in the browser (e.g. http://localhost:8501).

The app loads the trained model, scaler, and feature config from [Hugging Face](https://huggingface.co/shiavm006/Crop-yield_pridiction) — no need to store large `.pkl` files in this repo.

## Inputs & output

- **Input:** Area (country/region), Crop (item), Year, Rainfall (mm/year), Pesticides (tonnes), Average temperature (°C).
- **Output:** Predicted yield in **hg/ha** (hectogram per hectare).

## Dataset

- `Dataset/yield_df.csv` — used for dropdown options (Area, Item) and aligns with the model’s training data.