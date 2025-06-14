import xgboost as xgb
import joblib

# Load model
model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

# Load encoder
encoder = joblib.load("encoder.joblib")

import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# ================== SETUP STYLE ==================
st.set_page_config(page_title="Prediksi Harga Mobil Toyota", layout="centered")

# Custom CSS untuk UI navy-kuning
st.markdown("""
    <style>
        body {
            background-color: #0a1f44;
            color: #ffdd00;
        }
        .stApp {
            background-color: #0a1f44;
        }
        .css-1d391kg {  /* title */
            color: #ffdd00 !important;
        }
        h1, h2, h3, h4 {
            color: #ffdd00 !important;
        }
        label, .stSelectbox label, .stNumberInput label {
            color: #ffffff !important;
        }
        .stButton>button {
            background-color: #ffdd00;
            color: #0a1f44;
            font-weight: bold;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #f8e473;
            color: #000;
        }
    </style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL & DATA ==================
model = pickle.load(open('prediksi_hargamobil.sav', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
with open('metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# ================== TITLE & IMAGE ==================
st.title('🚗 Prediksi Harga Mobil Toyota Bekas')
st.image('mobil.png', use_container_width=True)

# ================== INPUT FORM ==================
with st.container():
    st.subheader("📋 Masukkan Data Mobil")

    car_models = sorted(list(set(pd.read_csv('toyota.csv')['model'].unique())))
    selected_model = st.selectbox('Model Mobil', car_models)

    transmissions = ['Manual', 'Automatic', 'Semi-Auto']
    selected_transmission = st.selectbox('Transmisi', transmissions)

    fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Other']
    selected_fuel_type = st.selectbox('Jenis Bahan Bakar', fuel_types)

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input('Tahun Produksi', min_value=2001, max_value=2024, step=1)
    with col2:
        mileage = st.number_input('Jarak Tempuh (KM)', min_value=0)

# ================== FORMAT ==================
def format_price(number):
    return "{:,.0f}".format(number).replace(",", ".")

# ================== PREDIKSI ==================
if st.button('🔮 Prediksi Harga'):
    if year == 0 or mileage == 0:
        st.warning('⚠️ Mohon lengkapi semua data input!')
    else:
        with st.spinner('🚀 Memproses prediksi...'):
            try:
                # Prepare categorical features
                categorical_features = pd.DataFrame({
                    'model': [selected_model],
                    'transmission': [selected_transmission],
                    'fuelType': [selected_fuel_type]
                })

                # Encode categorical features
                encoded_categorical = encoder.transform(categorical_features)

                # Prepare numerical features
                numerical_features = np.array([[year, mileage]])

                # Combine features
                prediction_input = np.hstack((numerical_features, encoded_categorical))

                # Make prediction
                prediction = model.predict(prediction_input)[0]

                # Convert to Rupiah
                prediction_rupiah = prediction * 19500

                # Display prediction
                st.success('✅ Prediksi Selesai!')
                st.subheader('💰 Prediksi Harga Mobil (IDR):')
                st.markdown(f"<h2 style='color:#ffdd00;'>Rp {format_price(prediction_rupiah)}</h2>", unsafe_allow_html=True)

                # Display metrics
                st.write("---")
                st.subheader("📊 Evaluasi Model")
                st.write(f"🔹 Mean Absolute Error (MAE): **{metrics['mae']:.2f}**")
                st.write(f"🔹 Mean Absolute Percentage Error (MAPE): **{metrics['mape']:.2f}%**")
                st.write(f"🔹 Akurasi Model: **{metrics['accuracy']:.2f}%**")

            except Exception as e:
                st.error(f'❌ Terjadi kesalahan: {str(e)}')
