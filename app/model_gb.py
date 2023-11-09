# model_gb_prediction.py

import pandas as pd
from joblib import load
import datetime
import sys
import streamlit as st
sys.path.append("./src/features")


def model_gb_prediction(inf_df):
    
    # Load the pre-trained Gradient Boosting model
    gradient_boosting_model = load('../models/Varun/gradient_boosting_model.joblib')

    # Convert 'flightDate' and 'departure_time' from string to datetime objects
    inf_df['flightDate'] = pd.to_datetime(inf_df['flightDate'])
    inf_df['departure_time'] = pd.to_datetime(inf_df['departure_time']).dt.time

    # Extract date and time features
    inf_df['flight_month'] = inf_df['flightDate'].dt.month
    inf_df['flight_day'] = inf_df['flightDate'].dt.day
    inf_df['flight_day_of_week'] = inf_df['flightDate'].dt.weekday
    inf_df['flight_week_of_year'] = inf_df['flightDate'].dt.isocalendar().week
    inf_df['flight_is_weekend'] = inf_df['flight_day_of_week'] >= 5
    if not pd.api.types.is_datetime64_any_dtype(inf_df['departure_time']):
        inf_df['departure_time'] = pd.to_datetime(inf_df['departure_time'], format='%H:%M:%S').dt.time
    inf_df['departure_hour'] = inf_df['departure_time'].apply(lambda x: x.hour if pd.notnull(x) else '')

    # Define departure time of day based on the hour
    def get_departure_time_of_day(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'

    inf_df['departure_time_of_day'] = inf_df['departure_hour'].apply(get_departure_time_of_day)

    # Select and rename columns as per the model's expected input
    model_input_df = inf_df[[
        'startingAirport', 'destinationAirport', 'flight_month', 'flight_day',
        'flight_day_of_week', 'flight_week_of_year', 'flight_is_weekend',
        'cabin_type', 'departure_time_of_day', 'travelDuration', 'totalDistance'
    ]].rename(columns={
        'startingAirport': 'startingAirport',
        'destinationAirport': 'destinationAirport',
        'flight_month': 'flightMonth',
        'flight_day': 'flightDay',
        'flight_day_of_week': 'flightDayOfWeek',
        'flight_week_of_year': 'flightWeekOfYear',
        'flight_is_weekend': 'flightIsWeekend',
        'cabin_type': 'cabin_type',
        'departure_time_of_day': 'departure_time',
        'travelDuration': 'traveltime_hours',
        'totalDistance': 'totalTravelDistance'

    })


    # Perform prediction with the loaded Gradient Boosting model
    prediction = gradient_boosting_model.predict(model_input_df)
    return prediction[0]

