import sys
import pandas as pd
import streamlit as st
sys.path.append("./src/features")

def model_tf_prediction(inf_df):
    from build_features import get_date_features, get_time_features
    inf_df = get_date_features(inf_df)
    inf_df = get_time_features(inf_df)

    import joblib
    from tensorflow.keras.models import load_model

    # Load the trained model
    model = load_model("./models/Kritika/exp2_best_model.h5")
    if model:
        st.write('Tensorflow prediction model has been loaded successfully!')

    # Load label encoders and scaler
    label_encoder = joblib.load('./models/Kritika/label_encoder.joblib')
    scaler = joblib.load('./models/Kritika/standard_scaler.joblib')

    # Apply label encoding on categorical columns in the new data
    for col in ['startingAirport', 'destinationAirport', 'cabin_type', 'departure_time_category']:
        inf_df[col] = label_encoder[col].transform(inf_df[col])

    # Scale the new data using the scaler fitted on the training data
    df_scaled = scaler.transform(inf_df)

    # Make prediction using the loaded model
    prediction = model.predict(df_scaled)
    pred_fare = '{:.2f}'.format(prediction[0][0])
    return pred_fare