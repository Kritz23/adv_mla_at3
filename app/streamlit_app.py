import streamlit as st
import pandas as pd
import os

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
    from model_tf import model_tf_prediction 

    predictions = {
        'TensorFlow': model_tf_prediction(inf_df),
        #'Catboost': model_catboost_prediction(inf_df),
        #'Model 3': model_custom1_prediction(inf_df),
        #'Model 4': model_custom2_prediction(inf_df)
    }
    st.write(f'Predicted fares for your {origin} to {destination} trip on {departure_date}:')

    # Display predictions in a table
    predictions_df = pd.DataFrame(predictions.items(), columns=['Model', 'Predicted Fare'])
    st.write(predictions_df)