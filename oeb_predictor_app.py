# -*- coding: utf-8 -*-
"""
OEB Prediction Pro - Complete UI Version
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from urllib.parse import quote
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from scipy.special import softmax
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
CNN_MODEL_NAME = "cnn_model_tf213_compatiblev2"  # Directory containing SavedModel files

def get_model_path(filename):
    """Constructs an absolute path to the model file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, MODEL_DIR, filename)

@st.cache_resource
def load_models_and_scalers():
    """Loads all models and scalers with TF SavedModel support."""
    scalers = {}
    classifiers = {}
    cnn_model = None
    
    try:
        # Load scalers
        scalers = {
            "desc": joblib.load(get_model_path("scaler_descriptors.pkl")),
            "cnn_input": joblib.load(get_model_path("scaler_features_cnn.pkl")),
            "cnn_output": joblib.load(get_model_path("scaler_features_cnn_output.pkl"))
        }
        
        # Load classifiers
        classifiers = {name: joblib.load(get_model_path(f"model_{name}.pkl")) for name in MODEL_NAMES}
        
        # Load CNN model in SavedModel format
        try:
            imported = tf.saved_model.load(get_model_path(CNN_MODEL_NAME))
            cnn_model = imported.signatures["serving_default"]
        except Exception as e:
            st.error(f"Failed to load CNN model: {str(e)}")
            return None, {}, {}
                
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
        # Calculate descriptors
        desc_calc = MolecularDescriptorCalculator(DESC_NAMES)
        descriptors = np.array(desc_calc.CalcDescriptors(mol))
        
        # Pad descriptors to 1024
        padded_desc = np.zeros(1024)
        actual_desc_len = len(descriptors)
        padded_desc[:min(actual_desc_len, 1024)] = descriptors[:min(actual_desc_len, 1024)]
        norm_desc = scalers["desc"].transform([padded_desc])[0]

        # Calculate fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_as_numpy_array = np.zeros((1024,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, fp_as_numpy_array)

        # Combine features
        combined_features = np.stack((norm_desc, fp_as_numpy_array), axis=-1)
        
        if combined_features.shape != (1024, 2):
            st.error(f"Unexpected shape for combined_features before reshape: {combined_features.shape}. Expected (1024,2).")
            return None
            
        # Reshape for CNN input
        cnn_input_image = combined_features.reshape(32, 32, 2)
        norm_input_flat = scalers["cnn_input"].transform(cnn_input_image.reshape(1, -1))
        norm_input_reshaped = norm_input_flat.reshape(1, 32, 32, 2)

        # Convert to tensor for SavedModel
        input_tensor = tf.convert_to_tensor(norm_input_reshaped, dtype=tf.float32)
        
        # Get predictions from SavedModel
        output = cnn_model(input_tensor)
        
        # Extract features from output dictionary
        features = output['output_0'].numpy() if 'output_0' in output else list(output.values())[0].numpy()
        
        # Scale features
        cnn_features_scaled = scalers["cnn_output"].transform(features)
        
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

    # Initialize session state
    if 'smiles_input' not in st.session_state:
        st.session_state.smiles_input = DEFAULT_SMILES

    # Load models
    cnn_model, scalers, classifiers = load_models_and_scalers()

    if cnn_model is None or not scalers or not classifiers:
        st.error("Application cannot start due to model loading errors.")
        return

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    selected_model_name = st.sidebar.selectbox("Select Model", MODEL_NAMES, index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("OEB Reference Guide")
    for oeb_val, desc in OEB_DESCRIPTIONS.items():
        st.sidebar.markdown(f"**OEB {oeb_val}:** {desc.split(':')[1].split('(')[0].strip()}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("Note: This tool provides predictions based on machine learning models. Always verify results with experimental data.")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Compound Input")
        
        # PubChem search
        st.markdown("**Search PubChem by Name**")
        pubchem_name = st.text_input("Enter compound name (e.g., Aspirin)", key="pubchem_search")
        
        if pubchem_name:
            with st.spinner(f"Searching PubChem for '{pubchem_name}'..."):
                pubchem_url, retrieved_smiles = get_pubchem_data(pubchem_name)
            
            if retrieved_smiles:
                st.success(f"Found: {pubchem_name}")
                st.info(f"SMILES: {retrieved_smiles}")
                if st.button(f"Use this compound", key="use_pubchem"):
                    st.session_state.smiles_input = retrieved_smiles
            elif pubchem_name:
                st.warning("Compound not found in PubChem")

        # Direct SMILES input
        st.markdown("**Or enter SMILES directly**")
        smiles = st.text_input("SMILES string", 
                             value=st.session_state.smiles_input,
                             key="smiles_input",
                             help="Enter the SMILES notation of your compound")
        
        # Example/clear buttons
        col_ex, col_clr = st.columns(2)
        with col_ex:
            if st.button("Load Example (Aspirin)"):
                st.session_state.smiles_input = DEFAULT_SMILES
                st.rerun()
        with col_clr:
            if st.button("Clear Input"):
                st.session_state.smiles_input = ""
                st.rerun()

        # Prediction button
        if st.button("🚀 Predict OEB", type="primary", use_container_width=True):
            if not st.session_state.smiles_input:
                st.error("Please enter a SMILES string or search PubChem")
            else:
                with st.spinner("Calculating OEB prediction..."):
                    features = compute_cnn_ready_features(st.session_state.smiles_input, scalers, cnn_model)
                
                if features is None:
                    st.error("Invalid SMILES or error in calculation")
                else:
                    model = classifiers[selected_model_name]
                    
                    # Get probabilities
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(features)[0]
                    else: 
                        decision_scores = model.decision_function(features)
                        if decision_scores.ndim == 1:
                            probs = softmax(np.array([decision_scores.squeeze()]), axis=1)[0]
                        else:
                            probs = softmax(decision_scores, axis=1)[0]
                    
                    probs = normalize_probabilities(probs, len(OEB_DESCRIPTIONS))
                    pred_class = int(np.argmax(probs))

                    # Display results
                    st.success(f"Predicted OEB: **{pred_class}**")
                    st.markdown(f"**{OEB_DESCRIPTIONS.get(pred_class, 'Unknown')}**")
                    
                    # Probability distribution
                    st.subheader("Probability Distribution")
                    prob_df = pd.DataFrame({
                        "OEB": list(OEB_DESCRIPTIONS.keys()),
                        "Description": [d.split(":")[0] for d in OEB_DESCRIPTIONS.values()],
                        "Probability": probs
                    }).set_index("OEB")
                    
                    st.dataframe(
                        prob_df.style.format({"Probability": "{:.2%}"})
                              .bar(subset=["Probability"], color='#5fba7d'),
                        use_container_width=True
                    )

    with col2:
        st.subheader("Molecule Information")
        
        if st.session_state.smiles_input:
            st.code(st.session_state.smiles_input, language="text")
            
            # Basic molecule properties
            mol = Chem.MolFromSmiles(st.session_state.smiles_input)
            if mol:
                st.markdown("**Molecular Properties**")
                col_mw, col_fp = st.columns(2)
                with col_mw:
                    st.metric("Molecular Weight", f"{Descriptors.MolWt(mol):.2f}")
                with col_fp:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    st.metric("Fingerprint Bits", f"{len(fp.GetOnBits())}")
                
                # Display PubChem link if available
                if pubchem_name and pubchem_url:
                    st.markdown(f"[View on PubChem ↗]({pubchem_url})", unsafe_allow_html=True)
            else:
                st.warning("Invalid SMILES - cannot compute properties")
        else:
            st.info("Enter a SMILES string or search PubChem to see molecular information")

if __name__ == "__main__":
    if not DESC_NAMES and hasattr(Descriptors, '_descList'): 
        try:
            DESC_NAMES = [desc[0] for desc in Descriptors._descList]
        except Exception:
            pass 
    if not DESC_NAMES:
        st.error("Critical Error: RDKit descriptor names could not be initialized.")
    main()
