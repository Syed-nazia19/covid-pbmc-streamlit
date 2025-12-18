import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# Paths (RELATIVE ONLY)
# ======================
MODEL_PATH = "models/xgboost_model_70_30.pkl"
SCALER_PATH = "models/scaler_xgb_70_30.pkl"
TEMPLATE_PATH = "data/template_input_from_model.csv"

# ======================
# Load model + scaler
# ======================
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# Extract canonical 45 features
feature_cols = list(scaler.feature_names_in_)

# ======================
# Streamlit UI
# ======================
st.title("COVID PBMC Classifier (45-Gene Model)")
st.write("Upload a CSV with gene expression values (rows = samples).")
st.write("Your uploaded file will be aligned to the model's 45 genes.")

# ======================
# Template download
# ======================
try:
    template_df = pd.read_csv(TEMPLATE_PATH)
except:
    template_df = pd.DataFrame(columns=feature_cols)

st.download_button(
    label="Download Template (45 Genes)",
    data=template_df.to_csv(index=False),
    file_name="template_input_45_genes.csv",
    mime="text/csv"
)

# ======================
# File upload
# ======================
uploaded_file = st.file_uploader(
    "Upload your gene expression CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # Try reading with index
    try:
        df = pd.read_csv(uploaded_file, index_col=0)
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

    st.write(f"Uploaded shape: {df.shape}")

    provided = df.columns.tolist()

    missing = [g for g in feature_cols if g not in provided]
    extra = [g for g in provided if g not in feature_cols]

    if missing:
        st.warning(
            f"{len(missing)} missing genes will be filled with zero. "
            f"Example: {missing[:10]}"
        )

    if extra:
        st.info(
            f"{len(extra)} extra genes will be ignored. "
            f"Example: {extra[:10]}"
        )

    # ======================
    # Align features
    # ======================
    aligned = pd.DataFrame(0, index=df.index, columns=feature_cols)
    common = [c for c in feature_cols if c in df.columns]

    if common:
        aligned.loc[:, common] = df.loc[:, common]

    # ======================
    # Scale
    # ======================
    try:
        X_scaled = scaler.transform(aligned)
    except Exception as e:
        st.error(f"❌ Scaler error: {e}")
        st.stop()

    # ======================
    # Predict
    # ======================
    try:
        probs = model.predict_proba(X_scaled)[:, 1]
        preds = model.predict(X_scaled)
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.stop()

    # ======================
    # Results
    # ======================
    results = df.copy()
    results["pred_prob"] = probs
    results["pred_label"] = preds

    st.success("✅ Prediction complete!")
    st.dataframe(results)

    st.download_button(
        label="Download Predictions",
        data=results.to_csv(),
        file_name="predictions.csv",
        mime="text/csv"
    )
