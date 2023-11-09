import requests
import math
import streamlit as st

def get_airport_coordinates(iata_code):
    api_url = "https://airport-info.p.rapidapi.com/airport"
    headers = {
        "X-RapidAPI-Key": "0c5dbfe378msh03b09ece8c38642p14bcfejsne04b93000444",
        "X-RapidAPI-Host": "airport-info.p.rapidapi.com"
    }
    querystring = {"iata": iata_code}

    response = requests.get(api_url, headers=headers, params=querystring)
    data = response.json()

    latitude = data['latitude']
    longitude = data['longitude']

    return latitude, longitude

def haversine(coord1, coord2):
    R = 3958.8
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def calculate_flight_duration(distance, average_speed=550):
    duration = distance / average_speed
    return duration

def get_flight_details(origin, destination):
    try:
        coord1 = get_airport_coordinates(origin)
        coord2 = get_airport_coordinates(destination)

        total_distance = haversine(coord1, coord2)
        travel_duration = calculate_flight_duration(total_distance)

        return travel_duration, total_distance
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None, None