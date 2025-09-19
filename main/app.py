import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, 'model.pkl')
scaler_path = os.path.join(script_dir, 'scaler.pkl')


with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #1a1a1a, #333333);
        animation: gradientAnimation 15s ease infinite;
        background-size: 200% 200%;
        color: white;
    }

    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    h1 {
        color: #FF4B4B !important;
    }

    .stSelectbox > div[data-baseweb="select"] > div:first-child {
        background-color: #333333 !important;
        border-color: #FF4B4B;
        color: white;
        font-size: 16px;
    }
    .stSelectbox > div[data-baseweb="select"] input,
    .stSelectbox > div[data-baseweb="select"] span {
        color: white !important;
        -webkit-text-fill-color: white !important;
    }
    .stSidebar h2 {
        color: #28a745 !important;
    }
    .stSidebar div[data-testid="stExpanderHeader"],
    .stSidebar div[data-testid="stExpanderHeader"] p,
    .stSidebar div[data-baseweb="select"] span,
    .stSidebar div[data-testid="stExpanderHeader"] div {
        color: #FF4B4B !important;
        font-weight: bold !important;
    }
    .stSelectbox > div[data-baseweb="select"] div[role="listbox"] {
        background-color: #333333 !important;
    }
    .stSelectbox > div[data-baseweb="select"] div[role="option"] {
        color: white;
        font-size: 16px;
        padding: 8px 12px;
    }
    .stSelectbox > div[data-baseweb="select"] div[role="option"]:hover {
        background-color: #FF4B4B;
        color: white;
    }
    .stSelectbox > div[data-baseweb="select"] > div:first-child > div[data-testid="stSelectboxChevron"] svg {
        fill: #FF4B4B !important;
        opacity: 1 !important;
        display: block !important;
    }
    .stSelectbox > div[data-baseweb="select"] > div:first-child > div[data-testid="stSelectboxChevron"] svg path {
        fill: #FF4B4B !important;
        opacity: 1 !important;
        display: block !important;
    }
    svg[data-baseweb="icon"][title="open"] path {
        fill: white !important;
        opacity: 1 !important;
        display: block !important;
    }

    p, label, .stMarkdown {
        color: white !important;
    }

    .st-emotion-cache-1r6dm1f {
        background-color: #333333;
    }
    .st-emotion-cache-10q07pt {
        color: white;
    }
    .st-emotion-cache-1c7y2o2 {
        color: white;
    }
    .st-emotion-cache-1av5qce {
        color: white;
    }
    .st-emotion-cache-1av5qce p {
        color: white !important;
    }
    .st-emotion-cache-1av5qce svg {
        fill: white !important;
    }
    .st-emotion-cache-1av5qce div[data-testid="stExpanderHeader"] {
        color: #FF4B4B !important;
    }
    .st-emotion-cache-1av5qce div[data-testid="stExpanderHeader"] p {
        color: #FF4B4B !important;
    }
    .st-emotion-cache-1av5qce div[data-testid="stExpanderHeader"] span {
        color: #FF4B4B !important;
    }
    .st-emotion-cache-1av5qce div[data-testid="stExpanderHeader"] div {
        color: #FF4B4B !important;
    }

    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }

    .stDataFrame {
        color: white;
    }
    .stDataFrame table {
        background-color: #333333;
        color: white;
    }
    .stDataFrame th {
        background-color: #444444;
        color: white;
    }
    .stDataFrame td {
        background-color: #333333;
        color: white;
    }
    .stDataFrame tr:nth-child(even) td {
        background-color: #2a2a2a;
    }

    .stMetric {
        color: white;
    }
    .stMetric > div > div:first-child {
        color: #FF4B4B;
    }
    .stMetric > div > div:nth-child(2) {
        color: white;
    }

</style>
""", unsafe_allow_html=True)

st.write('This app predicts whether a customer is likely to churn based on their information.')

st.sidebar.header('Customer Information')

def user_input_features():
    with st.sidebar.expander("Demographic Information", expanded=True):
        gender = st.selectbox('Gender', ('Male', 'Female'))
        senior_citizen = st.selectbox('Senior Citizen', ('No', 'Yes'))
        partner = st.selectbox('Partner', ('No', 'Yes'))
        dependents = st.selectbox('Dependents', ('No', 'Yes'))
    
    with st.sidebar.expander("Core Services", expanded=True):
        phone_service = st.selectbox('Phone Service', ('No', 'Yes'))
        multiple_lines = st.selectbox('Multiple Lines', ('No', 'Yes', 'No phone service'))
        internet_service = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    
    with st.sidebar.expander("Add-on Services", expanded=True):
        online_security = st.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
        online_backup = st.selectbox('Online Backup', ('No', 'Yes', 'No internet service'))
        device_protection = st.selectbox('Device Protection', ('No', 'Yes', 'No internet service'))
        tech_support = st.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
        streaming_tv = st.selectbox('Streaming TV', ('No', 'Yes', 'No internet service'))
        streaming_movies = st.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service'))
    
    with st.sidebar.expander("Contract & Billing", expanded=True):
        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        paperless_billing = st.selectbox('Paperless Billing', ('No', 'Yes'))
        payment_method = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    
    with st.sidebar.expander("Billing Information", expanded=True):
        tenure = st.slider('Tenure (months)', 1, 72, 24)
        monthly_charges = st.slider('Monthly Charges', 18.0, 120.0, 70.0)
        total_charges = st.slider('Total Charges', 18.0, 8700.0, 1400.0)

    data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

def preprocess(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    train_cols = model.feature_names_in_
    
    for col in train_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_encoded = df_encoded[train_cols]

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
    
    return df_encoded

st.subheader('User Input')
styled_input_df = input_df.T.reset_index()
styled_input_df.columns = ['Feature', 'Value']
st.dataframe(styled_input_df, hide_index=True, use_container_width=True)

if st.button('Predict'):
    processed_df = preprocess(input_df)
    prediction = model.predict(processed_df)
    prediction_proba = model.predict_proba(processed_df)

    st.subheader('Prediction')
    with st.spinner('Calculating prediction...'):
        import time
        time.sleep(1)

        if prediction[0] == 1:
            st.error("💔 **The customer is likely to churn.**")
        else:
            st.success("✅ **The customer is likely to stay.**")

    st.subheader('Prediction Probability')
    churn_probability = prediction_proba[0][1]
    stay_probability = prediction_proba[0][0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Probability of Churn", value=f"{churn_probability:.2%}")
        st.progress(churn_probability)
    with col2:
        st.metric(label="Probability of Staying", value=f"{stay_probability:.2%}")
        st.progress(stay_probability)

    st.info("A higher probability of churn indicates a greater likelihood of the customer leaving.")