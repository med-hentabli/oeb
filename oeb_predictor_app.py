# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 14:51:14 2025
Updated and corrected by ChatGPT on request

@author: pc
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from urllib.parse import quote
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, Draw
from tensorflow.keras.models import load_model
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from scipy.special import softmax
from PIL import Image
import os

# --- CONFIG ---
st.set_page_config(page_title="OEB Prediction Pro", layout="wide", page_icon="🔬")

# --- CONSTANTS ---
MODEL_DIR = "models"
DESCRIPTOR_SIZE = 1024
MODEL_NAMES = ["MLP", "SVC", "XGBoost", "RandomForest", "DecisionTree"]
DEFAULT_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin

OEB_DESCRIPTIONS = {
    0: "No exposure limits: Minimal or no systemic toxicity.",
    1: "OEB 1: Low hazard (OEL: 1000 - 5000 µg/m³)",
    2: "OEB 2: Moderate hazard (OEL: 100 - 1000 µg/m³)",
    3: "OEB 3: High hazard (OEL: 10 - 100 µg/m³)",
    4: "OEB 4: Very high hazard (OEL: 1 - 10 µg/m³)",
    5: "OEB 5: Extremely high hazard (OEL: < 1 µg/m³)",
    6: "OEB 6: Extremely potent (OEL: < 0.1 µg/m³)"
}

try:
    DESC_NAMES = [desc[0] for desc in Descriptors._descList]
except Exception:
    DESC_NAMES = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors"]

# --- UTILITIES ---
def get_model_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_DIR, filename)

@st.cache_resource
def load_models_and_scalers():
    try:
        scalers = {
            "desc": joblib.load(get_model_path("scaler_descriptors.pkl")),
            "cnn_input": joblib.load(get_model_path("scaler_features_cnn.pkl")),
            "cnn_output": joblib.load(get_model_path("scaler_features_cnn_output.pkl"))
        }
        classifiers = {name: joblib.load(get_model_path(f"model_{name}.pkl")) for name in MODEL_NAMES}
        cnn_model = load_model(get_model_path("cnn_feature_extractor_model.h5"), compile=False)
        return cnn_model, scalers, classifiers
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, {}, {}

def compute_features(smiles, scalers, cnn_model):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not DESC_NAMES:
        return None
    desc_calc = MolecularDescriptorCalculator(DESC_NAMES)
    descriptors = np.array(desc_calc.CalcDescriptors(mol))
    padded = np.zeros(DESCRIPTOR_SIZE)
    padded[:min(len(descriptors), DESCRIPTOR_SIZE)] = descriptors[:min(len(descriptors), DESCRIPTOR_SIZE)]
    try:
        norm_desc = scalers["desc"].transform([padded])[0]
    except:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=DESCRIPTOR_SIZE)
    fp_array = np.zeros((DESCRIPTOR_SIZE,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    features = np.stack((norm_desc, fp_array), axis=-1)
    try:
        cnn_input = scalers["cnn_input"].transform(features.reshape(1, -1)).reshape(1, 32, 32, 2)
        cnn_output = cnn_model.predict(cnn_input)
        return scalers["cnn_output"].transform(cnn_output)
    except:
        return None

@st.cache_data(ttl=3600)
def get_pubchem_smiles(compound):
    try:
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(compound)}/cids/JSON"
        cid = requests.get(cid_url).json().get("IdentifierList", {}).get("CID", [None])[0]
        if cid:
            prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
            smiles = requests.get(prop_url).json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
            return f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}", smiles
    except:
        return None, None
    return None, None

def smiles_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        import io
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        return Image.open(io.BytesIO(png))
    return None

# --- MAIN APP ---
def main():
    st.title("🔬 OEB Prediction Pro")
    cnn_model, scalers, classifiers = load_models_and_scalers()
    if not cnn_model or not scalers or not classifiers:
        return

    st.sidebar.header("⚙️ Controls")
    model_name = st.sidebar.selectbox("Select Model", MODEL_NAMES)

    input_col, vis_col = st.columns([0.6, 0.4])
    with input_col:
        pubchem_name = st.text_input("Search PubChem (e.g. Aspirin)")
        pubchem_url, retrieved_smiles = get_pubchem_smiles(pubchem_name) if pubchem_name else (None, None)
        if retrieved_smiles and st.button("Use PubChem SMILES"):
            st.session_state.smiles = retrieved_smiles
        smiles = st.text_input("SMILES Input", st.session_state.get("smiles", DEFAULT_SMILES))
        st.session_state.smiles = smiles

        if st.button("🚀 Predict OEB"):
            features = compute_features(smiles, scalers, cnn_model)
            if features is None:
                st.error("Feature extraction failed.")
                return

            model = classifiers[model_name]
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                else:
                    scores = model.decision_function(features)
                    scores = np.array([scores]) if scores.ndim == 1 else scores
                    probs = softmax(scores, axis=1)[0]
            except:
                st.error("Prediction failed.")
                return

            if len(probs) != len(OEB_DESCRIPTIONS):
                st.warning("Mismatch between predicted probabilities and class definitions.")
                probs = np.full(len(OEB_DESCRIPTIONS), 1 / len(OEB_DESCRIPTIONS))

            pred_class = int(np.argmax(probs))
            st.success(f"Predicted OEB Class: {pred_class}")
            st.markdown(OEB_DESCRIPTIONS.get(pred_class, "Unknown"))

            prob_df = pd.DataFrame({
                "OEB Class": list(OEB_DESCRIPTIONS.keys()),
                "Description": [desc.split(":")[1].split("(")[0].strip() for desc in OEB_DESCRIPTIONS.values()],
                "Probability": probs
            }).set_index("OEB Class")
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}).bar("Probability", color="lightgreen"))

    with vis_col:
        img = smiles_to_image(smiles)
        if img:
            st.image(img, caption=smiles, use_column_width=True)
        if pubchem_url:
            st.markdown(f"[View on PubChem]({pubchem_url})")

    st.caption("Powered by Streamlit, RDKit, TensorFlow")

if __name__ == "__main__":
    main()
