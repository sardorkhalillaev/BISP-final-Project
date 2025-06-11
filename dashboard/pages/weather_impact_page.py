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

def get_weather_impact_inputs(df, key_prefix):
    inputs = {}
    st.subheader("🌦️ Weather Conditions for Impact Analysis")
    # Your 'WeatherCondition' column is categorical. If model uses this, selectbox is fine.
    # If model uses more granular weather (temp, wind), you'd add those.
    
    inputs['WeatherCondition_input'] = st.selectbox( # Renamed to avoid clash if 'WeatherCondition' is a direct feature
        "Select Overall Weather Condition", 
        options=get_options(df, 'WeatherCondition', default_val="Clear"), 
        key=f"{key_prefix}_weather_cond_select"
    )
    
    # Example: If your weather_estimator takes more specific inputs:
    st.markdown("_(Optional: Provide more specific weather details if your model uses them)_")
    col1, col2, col3 = st.columns(3)
    with col1:
        inputs['Temperature_C'] = st.number_input("Temperature (°C)", value=15.0, step=0.5, key=f"{key_prefix}_temp", help="If model uses specific temp.")
    with col2:
        inputs['WindSpeed_mph'] = st.number_input("Wind Speed (mph)", value=10.0, step=1.0, key=f"{key_prefix}_wind", help="If model uses specific wind.")
    with col3:
        inputs['Visibility_miles'] = st.slider("Visibility (miles)", min_value=0.0, max_value=10.0, value=7.0, step=0.1, key=f"{key_prefix}_vis", help="If model uses specific visibility.")
    
    # It might also need context of a flight
    st.subheader("✈️ Flight Context (Optional, if model is flight-specific)")
    flight_col1, flight_col2 = st.columns(2)
    with flight_col1:
        inputs['Context_Airline'] = st.selectbox("Context: Airline", options=get_options(df, 'Airline'), key=f"{key_prefix}_context_airline")
        inputs['Context_OriginAirport'] = st.selectbox("Context: Origin Airport", options=get_options(df, 'OriginAirport'), key=f"{key_prefix}_context_origin")
    with flight_col2:
        inputs['Context_ArrivalAirport'] = st.selectbox("Context: Destination Airport", options=get_options(df, 'ArrivalAirport'), key=f"{key_prefix}_context_dest")
        context_flight_date = st.date_input("Context: Flight Date", value=datetime.now().date(), key=f"{key_prefix}_context_date")
        context_flight_time = st.time_input("Context: Flight Time (Scheduled)", value=datetime.now().time(), key=f"{key_prefix}_context_time")
        inputs['Context_ScheduledTime_iso'] = datetime.combine(context_flight_date, context_flight_time).isoformat()
    
    return inputs

def render_page(df, model_helper, data_helper):
    st.header("🌦️ Weather Impact Analysis")
    MODEL_NAME = "weather_estimator"

    if not model_helper.is_model_loaded(MODEL_NAME):
        st.warning(f"Weather Impact model ('{MODEL_NAME}') is not loaded.")
        return
    if df is None or df.empty: # Needed for dropdowns
        st.info("Please load flight data for context and dropdown options.")
        return

    with st.form(key=f"{MODEL_NAME}_form"):
        raw_features = get_weather_impact_inputs(df, key_prefix=MODEL_NAME)
        submitted = st.form_submit_button("Analyze Weather Impact")

    if submitted:
        if raw_features.get('WeatherCondition_input') == "--Select--": # Add more critical fields
            st.error("Please select a weather condition.")
            return
        try:
            features_df = data_helper.preprocess_input_for_model(raw_features, MODEL_NAME)
            if features_df.empty:
                st.error("Preprocessing failed. Check DataHelper and input values.")
                return

            # The output of this model is highly dependent on how it was trained.
            # It could be a risk score, predicted delay component, categorical impact level, etc.
            prediction = model_helper.estimate_weather_impact(features_df) # Method in ModelHelper
            
            st.subheader("Weather Impact Assessment:")
            if isinstance(prediction, (float, int)):
                st.metric(label="Predicted Impact Score/Value", value=f"{prediction:.2f}")
                if prediction > 0.7: st.error("High potential weather impact.") # Example thresholds
                elif prediction > 0.4: st.warning("Moderate potential weather impact.")
                else: st.info("Low potential weather impact.")
            elif isinstance(prediction, str):
                st.info(f"**Assessed Impact:** {prediction}")
            elif isinstance(prediction, dict):
                for key, value in prediction.items():
                    st.write(f"  - **{key.replace('_', ' ').title()}:** {value}")
            elif prediction is not None and hasattr(prediction, '__iter__') and not isinstance(prediction, (str, bytes)):
                 st.write(f"**Impact Details:** {', '.join(map(str, prediction))}") # For list/array output
            else:
                st.write(f"Model output: {prediction}") # Generic fallback

        except KeyError as e:
            st.error(f"Feature mismatch error: {e}. Check input and DataHelper for '{MODEL_NAME}'.")
        except Exception as e:
            st.error(f"An error occurred during weather impact analysis: {e}")
            st.exception(e)