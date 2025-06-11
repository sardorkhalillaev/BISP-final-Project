
import joblib
import os
import streamlit as st
import numpy as np

class ModelHelper:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        
        self.model_files = {
            "delay_departure_regressor": "delay_time_estimation_dep.pkl",
            "delay_arrival_regressor": "delay_time_estimation_arr.pkl",
            "gate_classifier": "gate_classifier.pkl",
            "ontime_performance_classifier": "on_time_performance.pkl",
            "passenger_volume_forecaster": "passenger_volume_forecaster.pkl",
            "weather_estimator": "weather_estimation.pkl"
        }
        self._load_all_models_on_init()

    def _load_all_models_on_init(self):
        
        print("ModelHelper: Attempting to load ML models...")
        for model_key, filename in self.model_files.items():
            path = os.path.join(self.models_dir, filename)
            try:
                if os.path.exists(path):
                    self.models[model_key] = joblib.load(path)
                    print(f"ModelHelper: Successfully loaded model: {model_key}")
                else:
                    print(f"ModelHelper: Model file not found for {model_key}: {path}")
                    self.models[model_key] = None
            except Exception as e:
                print(f"ModelHelper: Error loading model {model_key} from {path}: {e}")
                self.models[model_key] = None

    def is_model_loaded(self, model_name):
        return self.models.get(model_name) is not None

    def get_model(self, model_name):
        return self.models.get(model_name)

    def predict_departure_delay(self, features_df):
        model = self.models.get("delay_departure_regressor")
        if model and not features_df.empty: return model.predict(features_df)
        if features_df.empty : print("Warning: Empty features_df passed to predict_departure_delay")
        return [np.nan] 

    def predict_arrival_delay(self, features_df):
        model = self.models.get("delay_arrival_regressor")
        if model and not features_df.empty: return model.predict(features_df)
        if features_df.empty : print("Warning: Empty features_df passed to predict_arrival_delay")
        return [np.nan]

    def predict_gate(self, features_df): 
        model = self.models.get("gate_classifier")
        if model and not features_df.empty: return model.predict(features_df)
        if features_df.empty : print("Warning: Empty features_df passed to predict_gate")
        return [np.nan] 

    def predict_on_time_performance(self, features_df):
        model = self.models.get("ontime_performance_classifier")
        if model and not features_df.empty:
            pred_class_encoded = model.predict(features_df)
            pred_proba = model.predict_proba(features_df)
            return pred_class_encoded, pred_proba
        if features_df.empty : print("Warning: Empty features_df passed to predict_on_time_performance")
        return ([np.nan], np.array([[np.nan, np.nan]])) 

    def forecast_passenger_volume(self, features_df): 
        model = self.models.get("passenger_volume_forecaster")
        if model and not features_df.empty:
            predictions = model.predict(features_df)
            return np.maximum(0, predictions) 
        if features_df.empty : print("Warning: Empty features_df passed to forecast_passenger_volume")
        return [np.nan] * (len(features_df) if not features_df.empty else 1)


    def estimate_weather_impact(self, features_df): 
        model = self.models.get("weather_estimator")
        if model and not features_df.empty: return model.predict(features_df)
        if features_df.empty : print("Warning: Empty features_df passed to estimate_weather_impact")
        return [np.nan]