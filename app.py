import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load saved models
models = {
    'Linear Regression': pickle.load(open('Linear_Regression_best_model.pkl', 'rb')),
    'Ridge Regression': pickle.load(open('Ridge_Regression_best_model.pkl', 'rb')),
    'Random Forest': pickle.load(open('Random_Forest_best_model.pkl', 'rb')),
    'XGBoost': pickle.load(open('XGBoost_best_model.pkl', 'rb'))
}

# Load user data
with open("users.json") as f:
    user_db = json.load(f)

# Custom CSS
def load_css():
    st.markdown("""
        <style>
        body {
            background-color: #f5f7fa;
        }
        .main {
            background: #ffffff;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 0px 20px rgba(0,0,0,0.1);
        }
        .stTextInput>div>div>input {
            border: 2px solid #4CAF50;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# Session state init
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 Login to Big Mart Sales Predictor")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if username in user_db and user_db[username] == password:
            st.session_state.logged_in = True
            st.success("✅ Login successful 🎉")
        else:
            st.error("❌ Invalid username or password")

def predictor_dashboard():
    st.title("🛒 Big Mart Sales Prediction Dashboard")

    # Logout button in sidebar
    with st.sidebar:
        st.write("👤 Logged in")
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()

    st.header("📋 Enter Item Details")

    item_weight = st.number_input("Item Weight", min_value=0.0, step=0.1)
    item_fat_content = st.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
    item_visibility = st.slider("Item Visibility", 0.0, 0.5, 0.05)
    item_type = st.selectbox("Item Type", ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household'])
    item_mrp = st.number_input("Item MRP", min_value=0.0)
    outlet_identifier = st.selectbox("Outlet Identifier", ['OUT049', 'OUT018', 'OUT010'])
    outlet_establishment_year = st.selectbox("Outlet Establishment Year", [1985, 1987, 1997, 2004, 2009])
    outlet_size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
    outlet_location_type = st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
    outlet_type = st.selectbox("Outlet Type", ['Supermarket Type1', 'Supermarket Type2', 'Grocery Store'])

    selected_model = st.selectbox("Choose a model", list(models.keys()))

    if st.button("🚀 Predict Sales"):
        input_data = pd.DataFrame({
            'Item_Weight': [item_weight],
            'Item_Fat_Content': [item_fat_content],
            'Item_Visibility': [item_visibility],
            'Item_Type': [item_type],
            'Item_MRP': [item_mrp],
            'Outlet_Identifier': [outlet_identifier],
            'Outlet_Establishment_Year': [outlet_establishment_year],
            'Outlet_Size': [outlet_size],
            'Outlet_Location_Type': [outlet_location_type],
            'Outlet_Type': [outlet_type]
        })

        model = models[selected_model]
        pred_log = model.predict(input_data)
        sales = np.exp(pred_log[0])

        st.success(f"💰 Predicted Sales: ₹{sales:.2f}")

# Route
if not st.session_state.logged_in:
    login_page()
else:
    predictor_dashboard()
