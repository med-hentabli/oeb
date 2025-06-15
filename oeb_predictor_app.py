# -*- coding: utf-8 -*-
"""
OEB Prediction Pro - Updated for TensorFlow SavedModel format
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
            st.success("Successfully loaded CNN model in SavedModel format")
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
        
        # Extract features from output dictionary (adjust key if needed)
        features = output['output_0'].numpy() if 'output_0' in output else list(output.values())[0].numpy()
        
        # Scale features
        cnn_features_scaled = scalers["cnn_output"].transform(features)
        
        return cnn_features_scaled
        
    except Exception as e:
        st.error(f"Error in feature computation: {e}")
        return None

# ... [rest of your existing functions remain unchanged until main()] ...

def main():
    st.title("🔬 OEB Prediction Pro")
    st.markdown("Predict Occupational Exposure Bands for chemical compounds using advanced machine learning models.")

    # Display TensorFlow version for debugging
    st.sidebar.markdown(f"**TensorFlow Version:** {tf.__version__}")
    
    cnn_model, scalers, classifiers = load_models_and_scalers()

    if cnn_model is None or not scalers or not classifiers:
        st.error("""
        Application cannot start due to model loading errors. Please check:
        1. All model files are in the 'models' directory
        2. The CNN model is in SavedModel format in directory: cnn_model_tf213_compatiblev2
        3. File permissions are correct
        """)
        st.markdown(f"Expected model directory: `{os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_DIR)}`")
        return

    # ... [rest of your existing main() function remains unchanged] ...

if __name__ == "__main__":
    if not DESC_NAMES and hasattr(Descriptors, '_descList'): 
        try:
            DESC_NAMES = [desc[0] for desc in Descriptors._descList]
        except Exception:
            pass 
    if not DESC_NAMES:
        st.error("Critical Error: RDKit descriptor names (DESC_NAMES) could not be initialized.")
    main()
