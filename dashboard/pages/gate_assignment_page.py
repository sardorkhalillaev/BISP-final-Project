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

def get_gate_assignment_inputs(df, key_prefix):
    inputs = {}
    st.subheader("✈️ Flight Details for Gate Assignment")

    col1, col2, col3 = st.columns(3)
    with col1:
        inputs['Airline'] = st.selectbox("Airline", options=get_options(df, 'Airline'), key=f"{key_prefix}_airline")
        inputs['OriginAirport'] = st.selectbox("Origin Airport", options=get_options(df, 'OriginAirport'), key=f"{key_prefix}_origin")
    with col2:
        inputs['ArrivalAirport'] = st.selectbox("Arrival Airport (This flight's destination)", options=get_options(df, 'ArrivalAirport'), key=f"{key_prefix}_dest")
        inputs['AircraftType'] = st.selectbox("Aircraft Type", options=get_options(df, 'AircraftType'), key=f"{key_prefix}_actype")
    with col3:
        inputs['GateType'] = st.selectbox("Preferred/Required Gate Type", options=get_options(df, 'GateType'), key=f"{key_prefix}_gatetype_pref")
        # PreviousFlightDelay might be relevant if it affects aircraft positioning
        inputs['PreviousFlightDelay'] = st.number_input("Previous Flight's Arrival Delay at THIS Airport (minutes)", value=0, step=1, key=f"{key_prefix}_prev_arr_delay_here")

    st.subheader("🕒 Scheduled Arrival at this Airport")
    arr_date = st.date_input("Scheduled Arrival Date", value=datetime.now().date(), key=f"{key_prefix}_arr_date")
    arr_time_val = st.time_input("Scheduled Arrival Time", value=datetime.now().time(), key=f"{key_prefix}_arr_time")
    inputs['ScheduledArrivalTime_iso'] = datetime.combine(arr_date, arr_time_val).isoformat()
    
    # Also consider Scheduled Departure time from this gate if the model uses turnaround
    st.subheader("🕒 Scheduled Departure from this Gate (if applicable)")
    dep_date_from_gate = st.date_input("Scheduled Departure Date from this Gate", value=datetime.now().date(), key=f"{key_prefix}_dep_gate_date")
    dep_time_from_gate_val = st.time_input("Scheduled Departure Time from this Gate", value=datetime.now().time(), key=f"{key_prefix}_dep_gate_time")
    inputs['ScheduledDepartureFromGateTime_iso'] = datetime.combine(dep_date_from_gate, dep_time_from_gate_val).isoformat()
    
    # inputs['AircraftTurnaroundTime'] = st.number_input("Planned Turnaround Time (min)", value=60, step=5, key=f"{key_prefix}_turnaround_gate")

    return inputs

def render_page(df, model_helper, data_helper):
    st.header("🚪 Gate Assignment Prediction")
    MODEL_NAME = "gate_classifier"

    if not model_helper.is_model_loaded(MODEL_NAME):
        st.warning(f"Gate Assignment model ('{MODEL_NAME}') is not loaded.")
        return
    if df is None or df.empty:
        st.info("Please load flight data for context and dropdown options.")
        return

    with st.form(key=f"{MODEL_NAME}_form"):
        raw_features = get_gate_assignment_inputs(df, key_prefix=MODEL_NAME)
        submitted = st.form_submit_button("Predict Gate")

    if submitted:
        if raw_features.get('Airline') == "--Select--": # Add more critical fields
            st.error("Please select all required flight details.")
            return
        try:
            features_df = data_helper.preprocess_input_for_model(raw_features, MODEL_NAME)
            if features_df.empty:
                st.error("Preprocessing failed. Check DataHelper and input values.")
                return

            prediction = model_helper.predict_gate(features_df) # Method in ModelHelper
            assigned_gate = prediction[0] # Assuming it returns the gate name/number
            
            st.success(f"**Predicted Gate: {assigned_gate}**")
            st.balloons()
        except KeyError as e:
            st.error(f"Feature mismatch error: {e}. Check input and DataHelper for '{MODEL_NAME}'.")
        except Exception as e:
            st.error(f"An error occurred during gate assignment: {e}")
            st.exception(e)