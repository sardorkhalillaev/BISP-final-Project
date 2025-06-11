import streamlit as st
import pandas as pd
import os
from datetime import datetime # Keep if used directly in main

# Import utilities
from utils.model_helper import ModelHelper
from utils.data_helper import DataHelper

# Import page rendering functions
from pages import (
    overview_page,
    departure_delay_page,
    arrival_delay_page, # Assuming you have an arrival_delay_page.py
    gate_assignment_page,
    ontime_performance_page,
    passenger_forecast_page,
    weather_impact_page
)

# --- CONFIGURATION AND INITIALIZATION ---
st.set_page_config(
    page_title="Airport Operations ML Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded" # Or "auto" or "collapsed"
)

# Path to your data file
DEFAULT_DATA_PATH = "data/TAS_Airport_Data_2024_cleaned_exploratory.csv"

def load_css():
    css_path = "styles/style.css"
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found at {css_path}. Using fallback/default styles.")

@st.cache_resource # Caching is good for these helpers
def initialize_helpers():
    """Initialize and return data and model helpers."""
    data_helper = DataHelper()
    model_helper = ModelHelper() # ModelHelper's __init__ should now load all models
    return data_helper, model_helper

# THIS IS THE KEY CHANGE: Note the underscore before data_helper_instance
@st.cache_data 
def load_app_data(data_path, _data_helper_instance): # <-- Underscore added here
    """Load the application's primary dataset."""
    if os.path.exists(data_path):
        try:
            with st.spinner(f"Loading flight data from {data_path}..."):
                # Use the passed-in instance, even though it's not part of the cache key
                df = _data_helper_instance.load_csv_data(data_path) 
            if df is not None and not df.empty:
                st.session_state.data_loaded_successfully = True
                return df
            else:
                st.session_state.data_loaded_successfully = False
                st.error(f"Failed to load data from {data_path} or the file is empty.")
                return None
        except Exception as e:
            st.session_state.data_loaded_successfully = False
            st.error(f"Error loading data from {data_path}: {e}")
            return None
    else:
        st.session_state.data_loaded_successfully = False
        st.error(f"Data file not found: {data_path}. Please ensure it exists.")
        return None

# Initialize session state for data loading status
if 'data_loaded_successfully' not in st.session_state:
    st.session_state.data_loaded_successfully = False
if 'models_checked' not in st.session_state: # To show model status once
    st.session_state.models_checked = False


# --- Main Application ---
def main():
    load_css()
    
    # Initialize helpers (ModelHelper loads models internally now)
    data_h, model_h = initialize_helpers()

    # Load data automatically
    # We pass data_h to load_app_data because it might have data loading logic
    app_df = load_app_data(DEFAULT_DATA_PATH, data_h)

    # Main header
    st.markdown(
        '<div class="main-header"><h1>🛫 TAS Airport Operations ML Dashboard</h1></div>',
        unsafe_allow_html=True
    )

    # --- SIDEBAR (Simplified) ---
    with st.sidebar:
        st.header("⚙️ System Status")
        
        if st.session_state.data_loaded_successfully:
            st.success(f"✅ Flight data loaded ({len(app_df)} records).")
        else:
            st.error("⚠️ Flight data could not be loaded. Dashboard may not function correctly.")

        st.markdown("---")
        st.subheader("ML Model Status:")
        
        
        model_keys_info = {
            "delay_departure_regressor": "Departure Delay",
            "delay_arrival_regressor": "Arrival Delay",
            "gate_classifier": "Gate Assignment",
            "ontime_performance_classifier": "On-Time Performance",
            "passenger_volume_forecaster": "Passenger Forecast",
            "weather_estimator": "Weather Impact"
        }
        all_models_loaded = True
        for key, name in model_keys_info.items():
            if model_h.is_model_loaded(key):
                st.markdown(f"<small> {name} model active.</small>", unsafe_allow_html=True)
            else:
                st.markdown(f"<small>{name} model failed to load.</small>", unsafe_allow_html=True)
                all_models_loaded = False
        
        if not all_models_loaded:
             st.warning("Some predictive models are not available. Predictions may be limited.")
        else:
             st.success("All predictive models are active.")
        st.session_state.models_checked = True # Mark as checked
        st.markdown("---")
        st.caption(f"Data Source: `{DEFAULT_DATA_PATH}`")
    # --- END SIDEBAR ---


    # --- MAIN CONTENT AREA ---
    if not st.session_state.data_loaded_successfully or app_df is None:
        st.error("Dashboard cannot proceed without flight data. Please check the data file and restart.")
        # You could add more detailed instructions here if needed
        return # Stop further execution if data isn't loaded

    # If models are critical for all tabs, you might also check all_models_loaded here
    # For now, individual tabs will check their specific model.

    tab_titles = [
        "📊 Overview", "⏱️ Arr. Delay", "⏱️ Dep. Delay", "🚪 Gate Assign.", 
        "✈️ On-Time Perf.", "👥 Pass. Forecast", "🌦️ Weather Impact"
    ]
    
    (tab_overview, tab_arr_delay, tab_dep_delay, tab_gate, 
     tab_ontime, tab_passenger, tab_weather) = st.tabs(tab_titles)

    with tab_overview:
        overview_page.render_page(app_df, data_h)

    with tab_arr_delay:
        # Assuming arrival_delay_page.py exists and is imported
        arrival_delay_page.render_page(app_df, model_h, data_h) 
    
    with tab_dep_delay:
        departure_delay_page.render_page(app_df, model_h, data_h)

    with tab_gate:
        gate_assignment_page.render_page(app_df, model_h, data_h)

    with tab_ontime:
        ontime_performance_page.render_page(app_df, model_h, data_h)
    
    with tab_passenger:
        passenger_forecast_page.render_page(app_df, model_h, data_h) 
        
    with tab_weather:
        weather_impact_page.render_page(app_df, model_h, data_h)

if __name__ == "__main__":
    main()