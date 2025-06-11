import streamlit as st
import pandas as pd
from datetime import datetime, time 


def get_options(df_source, column_name, default_val="--Select--", sort_options=True):
    options = [default_val]
    if df_source is not None and column_name in df_source.columns:
        unique_values = df_source[column_name].dropna().astype(str).str.strip().unique()
        valid_options = [opt for opt in unique_values if opt] 
        if valid_options:
            if sort_options: options.extend(sorted(list(valid_options)))
            else: options.extend(list(valid_options))
    if not options or len(options) == 1 and options[0] == default_val : return [default_val, "N/A (No data)"]
    return options

def get_departure_delay_inputs(df, key_prefix):
    inputs = {}
    st.subheader("✈️ Flight Details & Conditions for Departure Delay")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        inputs['Airline'] = st.selectbox("Airline", options=get_options(df, 'Airline'), key=f"{key_prefix}_airline")
        inputs['OriginAirport'] = st.selectbox("Origin Airport", options=get_options(df, 'OriginAirport'), key=f"{key_prefix}_origin") # DepartureAirport in your schema
    with col2:
        inputs['ArrivalAirport'] = st.selectbox("Destination Airport", options=get_options(df, 'ArrivalAirport'), key=f"{key_prefix}_dest")
        inputs['AircraftType'] = st.selectbox("Aircraft Type", options=get_options(df, 'AircraftType'), key=f"{key_prefix}_actype")
    with col3:
        inputs['PreviousFlightDelay'] = st.number_input("Previous Flight Delay (minutes)", value=0, step=1, key=f"{key_prefix}_prevdelay")
        inputs['GateType'] = st.selectbox("Gate Type", options=get_options(df, 'GateType'), key=f"{key_prefix}_gatetype")

    st.subheader("📅 Scheduled Departure Timing")
    dep_date = st.date_input("Scheduled Departure Date", value=datetime.now().date(), key=f"{key_prefix}_dep_date")
    dep_time_val = st.time_input("Scheduled Departure Time", value=datetime.now().time(), key=f"{key_prefix}_dep_time")
    inputs['ScheduledTime_iso'] = datetime.combine(dep_date, dep_time_val).isoformat() # For DataHelper to parse

    st.subheader("🌦️ Conditions & Resources")
    cond_col1, cond_col2, cond_col3 = st.columns(3)
    with cond_col1:
        inputs['WeatherCondition'] = st.selectbox("Weather at Departure", options=get_options(df, 'WeatherCondition'), key=f"{key_prefix}_weather_dep")
    with cond_col2:
        inputs['GateAvailability'] = st.selectbox("Gate Availability at Departure", options=get_options(df, 'GateAvailability'), key=f"{key_prefix}_gateavail_dep")
    with cond_col3:
        runway_delay_median_val = 15 # Default integer
        if df is not None and 'RunwayDelay' in df.columns and df['RunwayDelay'].notna().any():
            try:
            # Calculate median, then convert to int. Handle potential NaNs if all are NaN.
                median_calc = df['RunwayDelay'].median()
                if pd.notna(median_calc):
                    runway_delay_median_val = int(round(median_calc)) # Round before int conversion
            except Exception: # Catch any error during median calculation or conversion
                pass # Keep default runway_delay_median_val

        inputs['RunwayDelay'] = st.number_input(
            "Expected Runway Taxi-Out Time (min)", 
            value=runway_delay_median_val,  # Now guaranteed to be an int
            step=1,                         # int
            key=f"{key_prefix}_runway_delay_input", 
            help="Typical or expected runway congestion/taxi time."
        )

    return inputs

def render_page(df, model_helper, data_helper):
    st.header("⏱️ Departure Delay Prediction")
    MODEL_NAME = "delay_departure_regressor" # Key from main.py and ModelHelper

    if not model_helper.is_model_loaded(MODEL_NAME):
        st.warning(f"Departure Delay model ('{MODEL_NAME}') is not loaded. Please load models.")
        return
    if df is None or df.empty:
        st.info("Please load flight data from the sidebar to make predictions.")
        return

    with st.form(key=f"{MODEL_NAME}_form"):
        raw_features = get_departure_delay_inputs(df, key_prefix=MODEL_NAME)
        submitted = st.form_submit_button("🔮 Predict Departure Delay")

    if submitted:
        # Validate inputs 
        if raw_features.get('Airline') == "--Select--" or raw_features.get('OriginAirport') == "--Select--": 
            st.error("Please select all required flight details (Airline, Origin Airport, etc.).")
            return

        try:
            features_df = data_helper.preprocess_input_for_model(raw_features, MODEL_NAME)
            if features_df.empty:
                st.error("Preprocessing failed. Check DataHelper and input values.")
                return
                
            prediction = model_helper.predict_departure_delay(features_df) 
            delay_minutes = prediction[0]
            
            st.metric(label="Predicted Departure Delay", value=f"{delay_minutes:.0f} minutes")
            if delay_minutes > 15:
                st.warning("Flight is predicted to have a significant departure delay.")
            elif delay_minutes < -5 : 
                 st.info("Flight is predicted to depart early or on-time.")
            else:
                st.success("Flight is predicted to depart on-time or with a minor delay.")
        except KeyError as e:
            st.error(f"Feature mismatch error: Required feature {e} not found or generated. Check input form and DataHelper.preprocess_input_for_model for '{MODEL_NAME}'.")
        except Exception as e:
            st.error(f"An error occurred during departure delay prediction: {e}")
            st.exception(e)