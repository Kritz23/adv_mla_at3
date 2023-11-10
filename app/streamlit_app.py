import streamlit as st
import pandas as pd
# Loading additional flight data from external API
from flight_data_fetcher import get_flight_details
# import function from build_features python script
from model_tf import model_tf_prediction
from model_catboost import model_catboost_prediction
from model_gb import model_gb_prediction
#from model_xgb import model_xgb_prediction

st.title('Flight Fare Estimation App')
st.write('Enter your trip details below:')

origin = st.text_input('Origin Airport')
destination = st.text_input('Destination Airport')
departure_date = st.date_input('Departure Date')
departure_time = st.time_input('Departure Time')
cabin_type = st.selectbox('Cabin Type', ['coach', 'premium coach', 'first', 'business'])

origin = origin.upper()
destination = destination.upper()

new_data = {
    'startingAirport': origin,
    'destinationAirport': destination,
    'cabin_type': cabin_type,
    'flightDate': str(departure_date),
    'departure_time': str(departure_time)
}
inf_df = pd.DataFrame([new_data])

if st.button('Predict Fare'):
    # Get travelDuration and totalDistance from AeroAPI
    travel_duration, total_distance = get_flight_details(origin, destination)
    
    # Create a copy of inf_df and add travel_duration and total_distance
    inf_df_api = inf_df.copy()
    inf_df_api['travelDuration'] = [travel_duration]
    inf_df_api['totalDistance'] = [total_distance]
    inf_df_gb = inf_df_api.copy()

    predictions = {
        'TensorFlow': model_tf_prediction(inf_df),
        'Catboost': model_catboost_prediction(inf_df_api),
        'GradientBoost': model_gb_prediction(inf_df_gb),
        #'XGBoost': model_xgb_prediction(inf_df)
    }
    st.write(f'Predicted fares for your {origin} to {destination} trip on {departure_date}:')

    # Display predictions in a table
    predictions_df = pd.DataFrame(predictions.items(), columns=['Model', 'Predicted Fare'])
    predictions_df['Predicted Fare'] = predictions_df['Predicted Fare'].apply(lambda x: f"${x}")
    st.write(predictions_df)

    # Display the average of predicted fares
    #st.write(f"Your average Predicted Fare is ${average_fare:.2f}")
