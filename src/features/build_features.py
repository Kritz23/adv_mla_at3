import pandas as pd
import numpy as np

def get_date_features(flight_data):
    df = flight_data.copy()
    df['flightDate'] = pd.to_datetime(df['flightDate'])
    df['month'] = df['flightDate'].dt.month
    df['day'] = df['flightDate'].dt.day
    df['weekday'] = df['flightDate'].dt.dayofweek
    
    return df.drop(["flightDate"], axis=1)

def get_time_features(flight_data):
    flight_data['departure_time'] = pd.to_datetime(flight_data['departure_time'])
    flight_data['departure_time_sin'] = np.sin(2 * np.pi * flight_data['departure_time'].dt.hour / 24)
    flight_data['departure_time_cos'] = np.cos(2 * np.pi * flight_data['departure_time'].dt.hour / 24)

    bins = [0, 6, 12, 18, 24]
    labels = ['night', 'morning', 'afternoon', 'evening']
    flight_data['departure_time_category'] = pd.cut(flight_data['departure_time'].dt.hour, bins=bins, labels=labels, include_lowest=True)

    return flight_data.drop(['departure_time'], axis=1)
