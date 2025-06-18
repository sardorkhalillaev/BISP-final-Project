# BISP-final-Project



## Overview

This project is a Streamlit-based web application designed to provide insights and predictive analytics for airport operations. It leverages machine learning models to forecast various aspects of flight activities, helping operational staff make informed decisions, optimize resources, and enhance overall airport efficiency.

The dashboard currently offers predictions and analyses for:
*   **Departure Delays:** Predicts the potential delay for departing flights.
*   **Arrival Delays:** Predicts the potential delay for arriving flights.
*   **Gate Assignments:** Suggests an optimal gate for arriving flights.
*   **On-Time Performance:** Classifies if a flight is likely to be on-time or delayed.
*   **Passenger Volume Forecast:** Forecasts the total number of passengers expected at the airport on future dates.
*   **Weather Impact Assessment:** Estimates the potential impact of weather conditions on flight operations.

## Project Structure
airport_dashboard/
├── main.py # Main Streamlit application runner
├── styles/
│ └── style.css # Custom CSS for styling
├── utils/
│ ├── init.py
│ ├── model_helper.py # Loads and serves ML models
│ └── data_helper.py # Handles data loading and preprocessing for models
├── models/ # Stores trained .pkl model files
│ ├── delay_time_estimation_dep.pkl
│ ├── delay_time_estimation_arr.pkl
│ ├── gate_classifier.pkl
│ ├── passenger_volume_forecaster.pkl
│ ├── on_time_performance.pkl
│ └── weather_estimation.pkl
├── models_meta/ # Stores preprocessing artifacts (encoders, scalers, feature names)
│ ├── preprocessor_dep_delay.pkl
│ ├── feature_names_dep_delay.pkl
│ ├── label_encoder_gate.pkl
│ └── ... (other metadata files) ...
├── data/
│ └── flights.csv # Primary dataset (e.g., TAS_Airport_Data_2024_cleaned_exploratory.csv)
├── pages/ # Contains Python files for each tab in the dashboard
│ ├── overview_page.py
│ ├── departure_delay_page.py
│ ├── arrival_delay_page.py
│ ├── gate_assignment_page.py
│ ├── ontime_performance_page.py
│ ├── passenger_forecast_page.py
│ └── weather_impact_page.py
models_training_crisp_dm/
├── Data_Understanding/ # Jupyter notebooks for model training and experimentation
│ ├── data_understanding.ipynb
│ ├── data_generation.ipynb
│ ├── data_balancing.ipynb
│ ├── EDA.ipynb
│ └── ... (other folders) ...
├── data/
│ └── flights.csv # Primary dataset (e.g., TAS_Airport_Data_2024_cleaned_exploratory.csv)
├── notebooks/ # Jupyter notebooks for model training and experimentation
│ ├── arrival_delay_modeling.ipynb
│ ├── data_understanding.ipynb
│ ├── gate_assignment_modelingg.ipynb
│ ├── ontime_performance_modeling.ipynb
│ ├── passenger_forecasting_modeling.ipynb
│ ├── weather_estimation_modeling.ipynb
│ └── departure_delay_modeling.ipynb
└── README.md # This file

## Features

*   **Interactive Dashboard:** User-friendly interface built with Streamlit.
*   **Data Overview:** Visualizations and KPIs summarizing historical flight data.
*   **Predictive Models:** Utilizes 6 distinct machine learning models for various operational aspects.
*   **Real-time Input (Simulated):** Users can input parameters for specific flights or scenarios to get predictions.
*   **Custom Styling:** A `style.css` file for a polished look and feel.
*   **Modular Code:** Separated logic for data handling, model serving, UI pages, and model training.

## Technology Stack

*   **Python:** Core programming language.
*   **Streamlit:** Web application framework for building the interactive dashboard.
*   **Pandas:** Data manipulation and analysis.
*   **NumPy:** Numerical computations.
*   **Scikit-learn:** Machine learning library for preprocessing, model training (used for some models/pipelines).
*   **XGBoost:** Gradient boosting library used for several predictive models.
    *   (Mention other specific ML libraries if used, e.g., RandomForest, Prophet)
*   **Joblib:** For saving and loading Python objects (models, preprocessors).
*   **Plotly / Seaborn / Matplotlib:** For data visualizations (used in notebooks and potentially in the dashboard).
*   **Jupyter Lab/Notebooks:** For model development and experimentation.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd airport_dashboard
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure your `requirements.txt` is up-to-date with all necessary packages and their versions, e.g., `streamlit`, `pandas`, `scikit-learn`, `xgboost`, `joblib`, `numpy`, `holidays`)*

4.  **Data and Models:**
    *   Ensure the `flights.csv` data file (e.g., `TAS_Airport_Data_2024_cleaned_exploratory.csv`) is placed in the `data/` directory.
    *   Ensure all trained model `.pkl` files are in the `models/` directory.
    *   Ensure all model metadata `.pkl` files (preprocessors, feature names, label encoders) are in the `models_meta/` directory.
        *(If these are large and not in Git, provide instructions on how to obtain or generate them).*

## Running the Dashboard

Once the setup is complete, run the Streamlit application from the `airport_dashboard` root directory:

```bash
streamlit run main.py