import streamlit as st
import pandas as pd
import sys
import os
sys.path.append("../src/features")

st.title('Flight Fare Estimation App')
st.write('Enter your trip details below:')

origin = st.text_input('Origin Airport')
destination = st.text_input('Destination Airport')
departure_date = st.date_input('Departure Date')
departure_time = st.time_input('Departure Time')
cabin_type = st.selectbox('Cabin Type', ['coach', 'premium coach', 'first', 'business'])

new_data = {
    'startingAirport': origin,
    'destinationAirport': destination,
    'cabin_type': cabin_type,
    'flightDate': str(departure_date),
    'departure_time': str(departure_time)
}
inf_df = pd.DataFrame([new_data])

if st.button('Predict Fare'):
    # import function from build_features python script 
    from build_features import get_date_features, get_time_features
    inf_df = get_date_features(inf_df)
    inf_df = get_time_features(inf_df)

    import joblib
    from tensorflow.keras.models import load_model

    # Load the trained model
    model = load_model("../models/Kritika/exp2_best_model.h5")
    if model:
        st.write('Trip Cost prediction model has been loaded successfully!')

    # Load label encoders and scaler
    label_encoder = joblib.load('../models/Kritika/label_encoder.joblib')
    scaler = joblib.load('../models/Kritika/standard_scaler.joblib')

    # Apply label encoding on categorical columns in the new data
    for col in ['startingAirport', 'destinationAirport', 'cabin_type', 'departure_time_category']:
        inf_df[col] = label_encoder[col].transform(inf_df[col])

    # Scale the new data using the scaler fitted on the training data
    df_scaled = scaler.transform(inf_df)

    # Make prediction using the loaded model
    prediction = model.predict(df_scaled)
    st.write(f'Predicted fare for your {origin} to {destination} trip is: {prediction[0][0]}')
