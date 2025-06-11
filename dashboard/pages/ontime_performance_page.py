import streamlit as st
import pandas as pd
from datetime import datetime, time

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
    # inputs['AircraftTurnaroundTime'] = st.number_input("Planned Turnaround Time (min)", value=60, step=5, key=f"{key_prefix}_turnaround_otp")
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
        if raw_features.get('Airline') == "--Select--": # Add more critical fields
            st.error("Please select all required flight details.")
            return
        try:
            features_df = data_helper.preprocess_input_for_model(raw_features, MODEL_NAME)
            if features_df.empty:
                st.error("Preprocessing failed. Check DataHelper and input values.")
                return
            
            # Assuming model_helper.predict_on_time_performance returns (class_prediction, probability_array)
            # e.g., (["On-Time"], [[0.2, 0.8]]) where 0.8 is prob of "On-Time"
            pred_class, pred_proba = model_helper.predict_on_time_performance(features_df)
            
            status = pred_class[0]
            # Find the probability of the predicted class. This needs careful handling
            # based on how your model_helper and model classes_ attribute are set up.
            # Example: if model.classes_ is ['Delayed', 'On-Time']
            try:
                # Assuming model_helper.get_model(MODEL_NAME).classes_ gives you the class order
                model_classes = model_helper.get_model_classes(MODEL_NAME) # You'll need to implement this in ModelHelper
                class_index = list(model_classes).index(status)
                probability = pred_proba[0][class_index] * 100
            except Exception: # Fallback if getting specific prob is complex
                probability = None


            st.subheader("Prediction Result:")
            if status.lower() == "on-time" or status.lower() == "ontime": # Adjust based on your model's output label
                st.success(f"**Predicted Status: {status}**")
                if probability is not None:
                    st.info(f"Confidence: {probability:.2f}%")
                st.balloons()
            else:
                st.error(f"**Predicted Status: {status}**")
                if probability is not None:
                    st.info(f"Confidence of this prediction: {probability:.2f}%")
        
        except KeyError as e:
            st.error(f"Feature mismatch error: {e}. Check input and DataHelper for '{MODEL_NAME}'.")
        except Exception as e:
            st.error(f"An error occurred during On-Time prediction: {e}")
            st.exception(e)