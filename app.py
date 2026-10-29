import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from train import FEATURES, train_and_save

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    """Load the model. If the pickle was made with a different scikit-learn
    version (or is missing), retrain it automatically and save a fresh copy."""
    try:
        pipe = pickle.load(open(BASE / 'pipe.pkl', 'rb'))
        df = pickle.load(open(BASE / 'df.pkl', 'rb'))
        # quick sanity check that the loaded model really works
        pipe.predict(df[FEATURES].head(1))
    except Exception:
        pipe, df = train_and_save(verbose=False)
    return pipe, df


pipe, df = load_artifacts()

st.title("💻 Laptop Price Predictor")

company = st.selectbox('Brand', sorted(df['Company'].unique()))
laptop_type = st.selectbox('Type', sorted(df['TypeName'].unique()))
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
weight = st.number_input('Weight of the laptop (kg)', min_value=0.5, max_value=5.0,
                         value=1.5, step=0.1)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen size (inches)', 10.0, 18.0, 15.6, step=0.1)
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', sorted(df['Cpu brand'].unique()))
hdd = st.selectbox('HDD (in GB)', [0, 32, 128, 500, 1000, 2000])
ssd = st.selectbox('SSD (in GB)', [0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 768, 1000, 1024], index=0)
gpu = st.selectbox('GPU', sorted(df['Gpu brand'].unique()))
os_name = st.selectbox('OS', sorted(df['os'].unique()))

if st.button('Predict Price'):
    x_res, y_res = (int(v) for v in resolution.split('x'))
    ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size

    query = pd.DataFrame([[
        company, laptop_type, ram, weight,
        1 if touchscreen == 'Yes' else 0,
        1 if ips == 'Yes' else 0,
        ppi, cpu, hdd, ssd, gpu, os_name
    ]], columns=FEATURES)

    price = int(np.exp(pipe.predict(query)[0]))
    st.subheader(f"The predicted price of this configuration is ₹ {price:,}")
