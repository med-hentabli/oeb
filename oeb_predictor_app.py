# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 14:51:14 2025
Updated by AI on Mon Jun 2 15:00:00 2025
Further Updated by AI on Mon Jun 2 20:07:00 2025

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
import io
import os
import tensorflow as tf

# --- PAGE CONFIG ---
st.set_page_config(page_title="OEB Prediction Pro", layout="wide", page_icon="🔬")

# --- CONSTANTS ---
try:
    DESC_NAMES = [desc[0] for desc in Descriptors._descList]
except AttributeError:
    st.warning("Could not dynamically load RDKit descriptor names. Using a predefined list might be necessary if errors occur.")
    DESC_NAMES = []

OEB_DESCRIPTIONS = {
    0: "No exposure limits: Minimal or no systemic toxicity.",
    1: "OEB 1: Low hazard (OEL: 1000 - 5000 µg/m³)",
    2: "OEB 2: Moderate hazard (OEL: 100 - 1000 µg/m³)",
    3: "OEB 3: High hazard (OEL: 10 - 100 µg/m³)",
    4: "OEB 4: Very high hazard (OEL: 1 - 10 µg/m³)",
    5: "OEB 5: Extremely high hazard (OEL: < 1 µg/m³)",
    6: "OEB 6: Extremely potent (OEL: < 0.1 µg/m³)"
}

MODEL_NAMES = ["MLP", "SVC", "XGBoost", "RandomForest", "DecisionTree"]
DEFAULT_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
MODEL_DIR = "models"

def get_model_path(filename):
    """Constructs an absolute path to the model file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, MODEL_DIR, filename)

@st.cache_resource
def load_models_and_scalers():
    """Loads all models and scalers."""
    scalers = {}
    classifiers = {}
    cnn_model = None
    try:
        scalers = {
            "desc": joblib.load(get_model_path("scaler_descriptors.pkl")),
            "cnn_input": joblib.load(get_model_path("scaler_features_cnn.pkl")),
            "cnn_output": joblib.load(get_model_path("scaler_features_cnn_output.pkl"))
        }
        classifiers = {name: joblib.load(get_model_path(f"model_{name}.pkl")) for name in MODEL_NAMES}
        cnn_model = load_model(get_model_path("cnn_feature_extractor_model.h5"), compile=False)
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Please ensure all model files are in the '{MODEL_DIR}' subdirectory.")
        return None, {}, {}
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        return None, {}, {}
    return cnn_model, scalers, classifiers

def compute_cnn_ready_features(smiles, scalers, cnn_model):
    """Computes features from SMILES string for CNN and ML models."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if not DESC_NAMES:
        st.error("Descriptor names (DESC_NAMES) are not defined. Cannot calculate descriptors.")
        return None
        
    try:
        desc_calc = MolecularDescriptorCalculator(DESC_NAMES)
        descriptors = np.array(desc_calc.CalcDescriptors(mol))
        
        padded_desc = np.zeros(1024)
        actual_desc_len = len(descriptors)
        padded_desc[:min(actual_desc_len, 1024)] = descriptors[:min(actual_desc_len, 1024)]
        norm_desc = scalers["desc"].transform([padded_desc])[0]

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_as_numpy_array = np.zeros((1024,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, fp_as_numpy_array)

        combined_features = np.stack((norm_desc, fp_as_numpy_array), axis=-1)
        
        if combined_features.shape != (1024, 2):
            st.error(f"Unexpected shape for combined_features before reshape: {combined_features.shape}. Expected (1024,2).")
            return None
            
        cnn_input_image = combined_features.reshape(32, 32, 2)
        norm_input_flat = scalers["cnn_input"].transform(cnn_input_image.reshape(1, -1))
        norm_input_reshaped = norm_input_flat.reshape(1, 32, 32, 2)

        if cnn_model is None:
            st.error("CNN model not loaded. Cannot extract features.")
            return None
            
        cnn_features_raw = cnn_model.predict(norm_input_reshaped)
        cnn_features_scaled = scalers["cnn_output"].transform(cnn_features_raw)
        
        return cnn_features_scaled
        
    except Exception as e:
        st.error(f"Error in feature computation: {e}")
        return None

@st.cache_data(ttl=3600)
def get_pubchem_data(compound_name):
    """Fetches compound CID and SMILES from PubChem."""
    if not compound_name:
        return None, None
    try:
        encoded_name = quote(compound_name)
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/cids/JSON"
        res_cid = requests.get(cid_url, timeout=10)
        res_cid.raise_for_status()
        cid = res_cid.json().get("IdentifierList", {}).get("CID", [None])[0]

        if cid:
            smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
            res_smiles = requests.get(smiles_url, timeout=10)
            res_smiles.raise_for_status()
            smiles = res_smiles.json().get("PropertyTable", {}).get("Properties", [{}])[0].get("CanonicalSMILES")
            pubchem_page_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
            return pubchem_page_url, smiles
    except requests.exceptions.RequestException as e:
        st.warning(f"PubChem API request failed: {e}")
    except Exception as e:
        st.warning(f"Error processing PubChem data for '{compound_name}': {e}")
    return None, None

def smiles_to_image(smiles, mol_size=(300, 300)):
    """Converts SMILES to a PIL Image of the molecule."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Draw.MolToImage(mol, size=mol_size)
    except Exception as e:
        st.error(f"Could not generate molecule image: {e}")
        return None

def normalize_probabilities(probs, target_length):
    """Normalizes probabilities to match target length."""
    if len(probs) == target_length:
        return probs
    
    normalized = np.zeros(target_length)
    common_len = min(len(probs), target_length)
    normalized[:common_len] = probs[:common_len]
    
    if np.sum(normalized) > 0:
        normalized = normalized / np.sum(normalized)
    else:
        normalized = np.full(target_length, 1/target_length if target_length > 0 else 1.0)
    
    return normalized

def main():
    st.title("🔬 OEB Prediction Pro")
    st.markdown("Predict Occupational Exposure Bands for chemical compounds using advanced machine learning models.")

    cnn_model, scalers, classifiers = load_models_and_scalers()

    if cnn_model is None or not scalers or not classifiers:
        st.error("Application cannot start due to model loading errors. Please check the console and ensure model files are correctly placed in the 'models' directory.")
        st.markdown(f"Expected model directory: `{os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_DIR)}`")
        return

    st.sidebar.header("⚙️ Controls & Options")
    selected_model_name = st.sidebar.selectbox("🤖 Choose Classifier Model", MODEL_NAMES, index=0)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 OEB Quick Reference")
    for oeb_val, desc in OEB_DESCRIPTIONS.items():
        st.sidebar.markdown(f"**OEB {oeb_val}:** {desc.split(':')[1].split('(')[0].strip()}")
    st.sidebar.markdown("---")
    st.sidebar.info("This app uses a CNN for feature extraction followed by a classical ML model for OEB prediction.")
    st.sidebar.caption("Developed with Streamlit, RDKit, TensorFlow, Scikit-learn.")

    input_col, vis_col = st.columns([0.6, 0.4])

    with input_col:
        st.subheader("🧪 Compound Input")
        st.markdown("**Option 1: Search PubChem by Name**")
        pubchem_name = st.text_input("Compound Name (e.g., Aspirin)", key="pubchem_name_input")
        
        retrieved_smiles = None
        pubchem_url = None
        if pubchem_name:
            with st.spinner(f"Searching PubChem for '{pubchem_name}'..."):
                pubchem_url, retrieved_smiles_from_api = get_pubchem_data(pubchem_name)
            if retrieved_smiles_from_api:
                st.success(f"Found '{pubchem_name}' on PubChem.")
                if st.button(f"Use SMILES for {pubchem_name}", key="use_pubchem_smiles"):
                    st.session_state.smiles_input = retrieved_smiles_from_api
                    st.rerun()
                retrieved_smiles = retrieved_smiles_from_api
            elif pubchem_url is None and retrieved_smiles_from_api is None and pubchem_name:
                st.warning(f"Could not find '{pubchem_name}' on PubChem or fetch its SMILES.")

        st.markdown("**Option 2: Enter SMILES String Directly**")
        if 'smiles_input' not in st.session_state:
            st.session_state.smiles_input = DEFAULT_SMILES
        smiles = st.text_input("SMILES String", value=st.session_state.smiles_input, key="smiles_text_input", 
                             help="Simplified Molecular Input Line Entry System")
        st.session_state.smiles_input = smiles

        col_ex, col_clear = st.columns(2)
        if col_ex.button("Load Example (Aspirin)", key="load_example"):
            st.session_state.smiles_input = DEFAULT_SMILES
            st.rerun()
        if col_clear.button("Clear SMILES", key="clear_smiles"):
            st.session_state.smiles_input = ""
            st.rerun()

        current_smiles_for_pred = retrieved_smiles if 'use_pubchem_smiles' in st.session_state and st.session_state.use_pubchem_smiles and retrieved_smiles else smiles
        
        if st.button("🚀 Predict OEB", type="primary", use_container_width=True):
            if not current_smiles_for_pred:
                st.error("❌ Please enter a SMILES string or find one via PubChem search.")
            else:
                with st.spinner("🔬 Analyzing molecule and predicting OEB... Please wait."):
                    features = compute_cnn_ready_features(current_smiles_for_pred, scalers, cnn_model)
                
                if features is None:
                    st.error("❌ Invalid SMILES string or error in feature computation. Please check the SMILES or model configurations.")
                else:
                    model = classifiers[selected_model_name]
                    
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(features)[0]
                    else: 
                        decision_scores = model.decision_function(features)
                        st.info("Model does not have `predict_proba`. Using `decision_function` with softmax.")
                        
                        if decision_scores.ndim == 1:
                            decision_scores_2d = np.array([decision_scores.squeeze()])
                            probs = softmax(decision_scores_2d, axis=1)[0]
                        elif decision_scores.ndim == 2:
                            probs = softmax(decision_scores, axis=1)[0]
                        else:
                            st.error(f"Decision scores have an unexpected shape: {decision_scores.shape}.")
                            probs = np.full(len(OEB_DESCRIPTIONS), 1/len(OEB_DESCRIPTIONS) if len(OEB_DESCRIPTIONS) > 0 else 1.0)

                    probs = normalize_probabilities(probs, len(OEB_DESCRIPTIONS))
                    pred_class = int(np.argmax(probs))

                    st.success(f"🎉 Predicted OEB Class: **{pred_class}**")
                    st.markdown(f"#### {OEB_DESCRIPTIONS.get(pred_class, 'Unknown OEB Class')}")
                    st.markdown("---")

                    st.subheader("📊 Probability Distribution")
                    prob_df_data = {
                        "OEB Class": list(OEB_DESCRIPTIONS.keys()),
                        "Description": [val.split(':')[1].split('(')[0].strip() for val in OEB_DESCRIPTIONS.values()],
                        "Probability": probs
                    }
                    prob_df = pd.DataFrame(prob_df_data).set_index("OEB Class")
                    st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}).bar(subset=["Probability"], color='lightgreen', vmin=0, vmax=1), 
                                  use_container_width=True)

    with vis_col:
        st.subheader("👁️ Molecule Viewer")
        current_smiles_for_vis = st.session_state.get('smiles_input', DEFAULT_SMILES)
        if current_smiles_for_vis:
            mol_image = smiles_to_image(current_smiles_for_vis)
            if mol_image:
                st.image(mol_image, caption=f"Structure for: {current_smiles_for_vis}", use_column_width=True)
            else:
                st.warning("Could not display molecule. SMILES might be invalid.")
        else:
            st.info("Enter a SMILES string or search PubChem to see the molecule structure.")

        if pubchem_name and pubchem_url: 
            st.markdown(f"🔗 [View **{pubchem_name}** on PubChem]({pubchem_url})", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("OEB Prediction Pro | Powered by AI and Cheminformatics")

if __name__ == "__main__":
    if not DESC_NAMES and hasattr(Descriptors, '_descList'): 
        try:
            DESC_NAMES = [desc[0] for desc in Descriptors._descList]
        except Exception:
            pass 
    if not DESC_NAMES:
        st.error("Critical Error: RDKit descriptor names (DESC_NAMES) could not be initialized. The application might not function correctly. Ensure RDKit is properly installed and accessible.")
    main()
