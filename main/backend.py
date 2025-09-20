import pandas as pd
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, 'model.joblib')
scaler_path = os.path.join(script_dir, 'scaler.joblib')


with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

def preprocess_and_predict(df):
    """
    Preprocesses the input DataFrame and makes a churn prediction.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    train_cols = model.feature_names_in_
    
    for col in train_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_encoded = df_encoded[train_cols]

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
    
    prediction = model.predict(df_encoded)
    prediction_proba = model.predict_proba(df_encoded)
    
    return prediction, prediction_proba
