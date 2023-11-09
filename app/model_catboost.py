import sys
import pandas as pd
import streamlit as st
import joblib
sys.path.append("./src/features")
from build_features import get_date_features

def model_catboost_prediction(inf_df):
    # Load the preprocessor and CatBoost model
    preprocessor = joblib.load('./models/Sahil/preprocessor.joblib')
    catboost_model = joblib.load('./models/Sahil/catboost_model.joblib')

    # if catboost_model:
    #     st.write('CatBoost prediction model has been loaded successfully!')

    # # Print the incoming dataframe
    # st.write('Incoming dataframe:')
    # st.write(inf_df)

    # Rename cabin_type feature to CabinClass
    inf_df.rename(columns={'cabin_type': 'CabinClass'}, inplace=True)

    # Run the df through get_date_features
    inf_df = get_date_features(inf_df)

    # # Print the incoming dataframe
    # st.write('Features dataframe:')
    # st.write(inf_df)

    # Preprocess the new data
    inf_df_preprocessed = preprocessor.transform(inf_df)

    # # Print the preprocessed dataframe
    # st.write('Preprocessed dataframe:')
    # st.write(inf_df_preprocessed)

    # Make prediction using the loaded model
    prediction = catboost_model.predict(inf_df_preprocessed)
    pred_fare = '{:.2f}'.format(prediction[0])
    return pred_fare