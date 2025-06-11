import streamlit as st
import pandas as pd
from datetime import datetime, time
import numpy as np

# --- Helper function (define or import) ---
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

def get_ontime_performance_inputs(df, key_prefix):
    # This will be very similar to delay prediction inputs
    inputs = {}
    st.subheader("✈️ Flight Details & Conditions for On-Time Prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        inputs['Airline'] = st.selectbox("Airline", options=get_options(df, 'Airline'), key=f"{key_prefix}_airline")
        inputs['OriginAirport'] = st.selectbox("Origin Airport", options=get_options(df, 'OriginAirport'), key=f"{key_prefix}_origin")
    with col2:
        inputs['ArrivalAirport'] = st.selectbox("Destination Airport", options=get_options(df, 'ArrivalAirport'), key=f"{key_prefix}_dest")
        inputs['AircraftType'] = st.selectbox("Aircraft Type", options=get_options(df, 'AircraftType'), key=f"{key_prefix}_actype")
    with col3:
        inputs['PreviousFlightDelay'] = st.number_input("Previous Flight Delay (minutes)", value=0, step=1, key=f"{key_prefix}_prevdelay")
        inputs['GateType'] = st.selectbox("Gate Type", options=get_options(df, 'GateType'), key=f"{key_prefix}_gatetype")

    st.subheader("📅 Scheduled Timings (Arrival Focus)") # On-Time usually refers to arrival
    arr_col1, arr_col2 = st.columns(2)
    with arr_col1:
        arr_date = st.date_input("Scheduled Arrival Date", value=datetime.now().date(), key=f"{key_prefix}_arr_date")
    with arr_col2:
        arr_time_val = st.time_input("Scheduled Arrival Time", value=datetime.now().time(), key=f"{key_prefix}_arr_time")
    inputs['ScheduledArrivalTime_iso'] = datetime.combine(arr_date, arr_time_val).isoformat()
    
    # Also include Scheduled Departure Time if your model uses it
    st.caption("Scheduled Departure Time (if relevant for model)")
    dep_col1, dep_col2 = st.columns(2)
    with dep_col1:
        dep_date = st.date_input("Scheduled Departure Date", value=datetime.now().date(), key=f"{key_prefix}_dep_date")
    with dep_col2:
        dep_time_val = st.time_input("Scheduled Departure Time", value=datetime.now().time(), key=f"{key_prefix}_dep_time")
    inputs['ScheduledTime_iso'] = datetime.combine(dep_date, dep_time_val).isoformat()


    st.subheader("🌦️ Conditions & Resources")
    cond_col1, cond_col2, cond_col3 = st.columns(3)
    with cond_col1:
        inputs['WeatherCondition'] = st.selectbox("Weather at Arrival", options=get_options(df, 'WeatherCondition'), key=f"{key_prefix}_weather_arr")
    with cond_col2:
        inputs['GateAvailability'] = st.selectbox("Gate Availability at Arrival", options=get_options(df, 'GateAvailability'), key=f"{key_prefix}_gateavail_arr")
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
    st.header("✈️ On-Time Performance Prediction")
    MODEL_NAME = "ontime_performance_classifier"

    if not model_helper.is_model_loaded(MODEL_NAME):
        st.warning(f"On-Time Performance model ('{MODEL_NAME}') is not loaded.")
        return
    if df is None or df.empty:
        st.info("Please load flight data for context and dropdown options.")
        return

    with st.form(key=f"{MODEL_NAME}_form"):
        raw_features = get_ontime_performance_inputs(df, key_prefix=MODEL_NAME)
        submitted = st.form_submit_button("Predict On-Time Performance")

    if submitted:
        try:
            features_df = data_helper.preprocess_input_for_model(raw_features, MODEL_NAME)
            if features_df.empty:
                st.error("Preprocessing failed. Check input values and DataHelper logs.")
                return

            prediction_encoded_array, prediction_proba_array = model_helper.predict_on_time_performance(features_df)
            
            single_prediction_encoded = prediction_encoded_array[0] # This is the numerical label, e.g., 0 or 1
            single_prediction_proba = prediction_proba_array[0]

            if pd.isna(single_prediction_encoded):
                st.warning("Could not predict on-time status. Input data might be insufficient.")
                return

            ontime_label_encoder = data_helper.label_encoders.get(MODEL_NAME)
            predicted_status_label = "Unknown Status" # Default string

            if ontime_label_encoder:
                # Ensure single_prediction_encoded is in a list or array for inverse_transform
                predicted_status_label_array = ontime_label_encoder.inverse_transform(np.array([single_prediction_encoded]))
                predicted_status_label = predicted_status_label_array[0] # This is now a STRING, e.g., "On-Time" or "Delayed"
                
                st.subheader("Prediction Result:")
                
                # vvvvvvvvvvvvvvvv USE THE STRING LABEL HERE vvvvvvvvvvvvvvvvv
                if predicted_status_label.lower() == "on-time": # Using the string label
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    st.success(f"**Predicted Status: {predicted_status_label}**")
                    st.balloons()
                else:
                    st.error(f"**Predicted Status: {predicted_status_label}**")

                # Display probabilities (this part seemed okay)
                try:
                    # Ensure single_prediction_encoded is treated as an integer index
                    proba_of_predicted_class = single_prediction_proba[int(single_prediction_encoded)] * 100
                    st.info(f"Confidence in this prediction: {proba_of_predicted_class:.2f}%")
                    
                    st.write("Probabilities for each status:")
                    for i, class_label_from_encoder in enumerate(ontime_label_encoder.classes_):
                        st.write(f"  - {class_label_from_encoder}: {single_prediction_proba[i]*100:.2f}%")
                except IndexError:
                    st.warning("Could not display detailed probabilities (IndexError).")
                except Exception as e_proba:
                    st.warning(f"Error displaying probabilities: {e_proba}")
            
            else:
                st.error(f"Label encoder for '{MODEL_NAME}' not found. Cannot display original status name. Predicted code: {single_prediction_encoded}")
            
        except KeyError as e:
            st.error(f"Feature mismatch error during On-Time prediction: {e}. Check input form and DataHelper for '{MODEL_NAME}'.")
            st.exception(e)
        except Exception as e:
            st.error(f"An error occurred during On-Time performance prediction: {e}")
            st.exception(e)