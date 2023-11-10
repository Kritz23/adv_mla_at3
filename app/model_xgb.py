#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import joblib
from joblib import load
import datetime
import sys
import streamlit as st
sys.path.append("./src/features")


def model_xgb_prediction(inf_df):
    from build_features import get_date_features, get_time_features
    inf_df = get_date_features(inf_df)
    inf_df = get_time_features(inf_df)
    
    # Load the pre-trained Gradient Boosting model
    xgb_model = load('./models/ANIKA/best_xgb_model_reg.joblib')
    """if xgb_model:
        st.write('XGBoost prediction model has been loaded successfully!')
    """
    # Check if 'departure_time_category' is present in the DataFrame
    if 'departure_time_category' in inf_df.columns:
        # Load label encoders and scaler
        label_encoder = joblib.load('./models/ANIKA/label_encoder.joblib')
        scaler = joblib.load('./models/ANIKA/standard_scaler.joblib')
        
        # Apply label encoding on categorical columns in the new data
        for col in ['startingAirport', 'destinationAirport', 'cabin_type', 'departure_time_category']:
            inf_df[col] = label_encoder[col].transform(inf_df[col])

        # Scale the new data using the scaler fitted on the training data
        df_scaled = scaler.transform(inf_df)

        # Make prediction using the loaded model
        prediction = xgb_model.predict(df_scaled)
        pred_fare = '{:.2f}'.format(prediction[0])
        return pred_fare
    else:
        st.write("'departure_time_category' column not found in the input DataFrame.")
        return None
