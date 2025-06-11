import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

def get_passenger_forecast_inputs(df, key_prefix): # df might be used for airport options if model is airport-specific
    inputs = {}
    st.subheader("🗓️ Forecasting Period & Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        inputs['start_date'] = st.date_input("Forecast Start Date", value=datetime.now().date() + timedelta(days=1), key=f"{key_prefix}_start_date")
    with col2:
        inputs['num_days'] = st.number_input("Number of Days to Forecast", min_value=1, max_value=90, value=7, key=f"{key_prefix}_num_days")

    # If your model is airport/terminal specific:
    # inputs['airport_code'] = st.selectbox("Airport for Forecast", options=get_options(df, 'OriginAirport'), key=f"{key_prefix}_airport")
    # inputs['terminal_id'] = st.text_input("Terminal (if applicable)", key=f"{key_prefix}_terminal")
    
    # You might also add inputs for known events, holidays if your model uses them
    # inputs['is_holiday_period'] = st.checkbox("Is this a holiday period?", key=f"{key_prefix}_holiday")
    return inputs

def render_page(df, model_helper, data_helper): # df might be used for context or dropdowns
    st.header("👥 Passenger Volume Forecast")
    MODEL_NAME = "passenger_volume_forecaster"

    if not model_helper.is_model_loaded(MODEL_NAME):
        st.warning(f"Passenger Forecasting model ('{MODEL_NAME}') is not loaded.")
        return
    # df might not be strictly needed for inputs if forecast is purely time-based,
    # but good to have for context or if model uses airport-specific historicals.

    with st.form(key=f"{MODEL_NAME}_form"):
        raw_params = get_passenger_forecast_inputs(df, key_prefix=MODEL_NAME)
        submitted = st.form_submit_button("Forecast Passengers")

    if submitted:
        try:
            start_date = raw_params['start_date']
            num_days = raw_params['num_days']
            
            # Prepare feature list for DataHelper for each day to forecast
            forecast_input_list = []
            forecast_dates = []
            for i in range(num_days):
                current_date = start_date + timedelta(days=i)
                forecast_dates.append(current_date)
                # DataHelper will need to convert this into features your model expects
                # (e.g., year, month, day, day_of_week, is_holiday, etc.)
                date_features = {
                    'date_iso': current_date.isoformat(), # For DataHelper to parse
                    # Add other parameters from raw_params if model needs them (e.g., airport_code)
                    # 'airport_code': raw_params.get('airport_code') 
                }
                forecast_input_list.append(date_features)

            # DataHelper processes a list of these raw daily feature dicts
            features_df_multi_day = data_helper.preprocess_input_for_model(forecast_input_list, MODEL_NAME)
            
            if features_df_multi_day.empty and num_days > 0 : # check if preprocessing failed
                 st.error("Preprocessing for passenger forecast failed. Check DataHelper.")
                 return

            predictions = model_helper.forecast_passenger_volume(features_df_multi_day) # Expects a DataFrame for multiple days

            if predictions is not None and len(predictions) == num_days:
                results_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecasted Passengers': [int(round(p)) for p in predictions] # Assuming numeric prediction
                })
                st.success("Passenger forecast generated:")
                st.dataframe(results_df.style.format({"Date": lambda t: t.strftime('%Y-%m-%d'), "Forecasted Passengers": "{:,}"}))
                
                fig = px.line(results_df, x='Date', y='Forecasted Passengers', markers=True,
                              title="Passenger Volume Forecast")
                fig.update_layout(xaxis_title="Date", yaxis_title="Number of Passengers")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Forecasting failed or returned unexpected number of results. Expected {num_days}, got {len(predictions) if predictions is not None else 'None'}.")
        except KeyError as e:
            st.error(f"Feature mismatch error: {e}. Check input and DataHelper for '{MODEL_NAME}'.")
        except Exception as e:
            st.error(f"An error occurred during passenger forecasting: {e}")
            st.exception(e)