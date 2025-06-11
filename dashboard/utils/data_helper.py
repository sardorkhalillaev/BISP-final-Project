# In utils/data_helper.py

import pandas as pd
import numpy as np
import streamlit as st # Keep for st.error/warning if needed during preprocessing
import os
from datetime import datetime
import joblib # Crucial for loading your .pkl files
import holidays
class DataHelper:
    def __init__(self):
        self.data = None # This will hold the main flights.csv data
        self.data_loaded = False

        # --- Load ALL Preprocessing Artifacts ---
        models_meta_dir = "models_meta" # Ensure this path is correct relative to main.py
        self.preprocessors = {}
        self.feature_names_lists = {} # To store the lists of expected feature names after preprocessing
        self.label_encoders = {}    # To store label encoders for classification targets

        model_configs = {
            "delay_departure_regressor": {"prefix": "dep_delay"},
            "delay_arrival_regressor": {"prefix": "arr_delay"},
            "gate_classifier": {"prefix": "gate", "has_label_encoder": True},
            "ontime_performance_classifier": {"prefix": "ontime", "has_label_encoder": True},
            "passenger_volume_forecaster": {"prefix": "passenger_daily_total"}, # Assuming daily total
            "weather_estimator": {"prefix": "weather_impact", "has_label_encoder": True} # Assuming weather impact is classification
        }

        for model_key, config in model_configs.items():
            prefix = config["prefix"]
            try:
                preprocessor_path = os.path.join(models_meta_dir, f"preprocessor_{prefix}.pkl")
                features_path = os.path.join(models_meta_dir, f"feature_names_{prefix}.pkl")

                if os.path.exists(preprocessor_path):
                    self.preprocessors[model_key] = joblib.load(preprocessor_path)
                else:
                    st.sidebar.warning(f"Preprocessor not found for {model_key} at {preprocessor_path}")
                    self.preprocessors[model_key] = None
                
                if os.path.exists(features_path):
                    self.feature_names_lists[model_key] = joblib.load(features_path)
                else:
                    st.sidebar.warning(f"Feature names list not found for {model_key} at {features_path}")
                    self.feature_names_lists[model_key] = None
                
                if config.get("has_label_encoder", False):
                    le_path = os.path.join(models_meta_dir, f"label_encoder_{prefix}.pkl")
                    if os.path.exists(le_path):
                        self.label_encoders[model_key] = joblib.load(le_path)
                    else:
                        st.sidebar.warning(f"Label encoder not found for {model_key} at {le_path}")
                        self.label_encoders[model_key] = None

            except Exception as e:
                st.sidebar.error(f"Error loading metadata for {model_key}: {e}")
                self.preprocessors[model_key] = None
                self.feature_names_lists[model_key] = None
                if config.get("has_label_encoder", False): self.label_encoders[model_key] = None
        
        # Feedback (optional)
        if any(p is not None for p in self.preprocessors.values()):
            st.sidebar.info("Some preprocessing objects loaded.")
        else:
            st.sidebar.error("No preprocessing objects were loaded. Predictions will likely fail.")

        # For Passenger Volume Forecaster (Daily Total), we need historical daily passenger data to create lags
        # This part is specific to the time series model.
        if "passenger_volume_forecaster" in self.preprocessors: # Check if its preprocessor loaded
            self._prepare_historical_daily_passengers()


    def _prepare_historical_daily_passengers(self):
        """
        Loads and prepares historical daily passenger data needed for lag features
        in the passenger_volume_forecaster.
        This should be called after the main flights.csv data is loaded.
        """
        self.historical_daily_passengers_ts = None
        if self.data is not None and not self.data.empty:
            temp_df = self.data.copy()
            if 'ScheduledTime' in temp_df.columns and 'Passengers' in temp_df.columns:
                temp_df['ScheduledTime'] = pd.to_datetime(temp_df['ScheduledTime'], errors='coerce')
                temp_df.dropna(subset=['ScheduledTime', 'Passengers'], inplace=True)
                temp_df['FlightDate'] = temp_df['ScheduledTime'].dt.date
                
                daily_passengers = temp_df.groupby('FlightDate')['Passengers'].sum().reset_index()
                daily_passengers.rename(columns={'Passengers': 'TotalDailyPassengers', 'FlightDate': 'Date'}, inplace=True)
                daily_passengers['Date'] = pd.to_datetime(daily_passengers['Date'])
                daily_passengers = daily_passengers.sort_values(by='Date').set_index('Date')
                
                # Reindex to ensure all dates are present (optional, but good for robust lag creation)
                # if not daily_passengers.empty:
                #     all_dates = pd.date_range(start=daily_passengers.index.min(), end=daily_passengers.index.max(), freq='D')
                #     daily_passengers = daily_passengers.reindex(all_dates)
                #     daily_passengers['TotalDailyPassengers'] = daily_passengers['TotalDailyPassengers'].interpolate(method='linear')
                
                self.historical_daily_passengers_ts = daily_passengers[['TotalDailyPassengers']] # Keep only target
                print("DataHelper: Historical daily passenger time series prepared.")
            else:
                print("DataHelper Warning: Could not prepare historical daily passengers due to missing columns.")
        else:
            print("DataHelper Warning: Main data not loaded, cannot prepare historical daily passengers.")


    @st.cache_data # Using _self as you did, which is fine for caching this method
    def load_csv_data(_self, data_path): # Changed _self to self (conventional)
        try:
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                if data.empty:
                    _self.data = None; _self.data_loaded = False
                    return pd.DataFrame() # Return empty DataFrame
                
                data.columns = data.columns.str.strip() # Good practice
                _self.data = data
                _self.data_loaded = True
                _self._prepare_historical_daily_passengers() # Prepare historical data after main load
                return data
            else:
                _self.data = None; _self.data_loaded = False
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading CSV data in DataHelper: {str(e)}")
            _self.data = None; _self.data_loaded = False
            return pd.DataFrame()


    def preprocess_input_for_model(self, raw_features_input, model_name):
        print(f"DataHelper: Preprocessing for model '{model_name}'")
        if isinstance(raw_features_input, dict):
            features_df = pd.DataFrame([raw_features_input])
        elif isinstance(raw_features_input, list) and model_name == "passenger_volume_forecaster": # Special handling for list input for passenger forecast
            features_df = pd.DataFrame(raw_features_input)
        elif isinstance(raw_features_input, list): # General case if other models might take lists (unlikely for now)
             features_df = pd.DataFrame(raw_features_input)
        else:
            st.error(f"Invalid input type for preprocessing: {type(raw_features_input)}")
            return pd.DataFrame()

        preprocessor = self.preprocessors.get(model_name)
        expected_feature_names_after_transform = self.feature_names_lists.get(model_name) # These are names AFTER transform

        if preprocessor is None:
            st.error(f"No preprocessor loaded for model: '{model_name}'. Cannot predict.")
            return pd.DataFrame()

        # --- Stage 1: Engineer features from raw inputs that the preprocessor itself doesn't handle ---
        # (e.g., parsing datetime strings from UI into numerical components like hour, month, dayofweek)
        # This step creates the columns that your preprocessor's ColumnTransformer was *trained* on.
        
        # Datetime engineering for departure/arrival based models
        datetime_iso_map = {
            "delay_departure_regressor": {"iso_col": "ScheduledTime_iso", "prefix": "Scheduled_"},
            "delay_arrival_regressor": {"iso_col": "ScheduledArrivalTime_iso", "prefix": "ScheduledArrival_"},
            "gate_classifier": {"iso_col": "ScheduledArrivalTime_iso", "prefix": "Arrival_"}, # Gate uses Arrival Time
            "ontime_performance_classifier": {"iso_col_arr": "ScheduledArrivalTime_iso", "prefix_arr": "SchArr_",
                                              "iso_col_dep": "ScheduledTime_iso", "prefix_dep": "SchDep_"}, # OnTime might use both
            "weather_estimator": {"iso_col": "Context_ScheduledTime_iso", "prefix": "WeatherObs_"} # If weather model uses flight time context
        }

        current_model_dt_config = datetime_iso_map.get(model_name)
        if current_model_dt_config:
            iso_cols_to_process = []
            if "iso_col" in current_model_dt_config: iso_cols_to_process.append(("iso_col", "prefix"))
            if "iso_col_arr" in current_model_dt_config: iso_cols_to_process.append(("iso_col_arr", "prefix_arr"))
            if "iso_col_dep" in current_model_dt_config: iso_cols_to_process.append(("iso_col_dep", "prefix_dep"))

            for iso_key, prefix_key in iso_cols_to_process:
                iso_col_name = current_model_dt_config[iso_key]
                prefix = current_model_dt_config[prefix_key]
                if iso_col_name in features_df.columns:
                    def parse_dt(iso_string):
                        if pd.isna(iso_string) or str(iso_string).lower() == "--select--":
                            return {f"{prefix}Hour": np.nan, f"{prefix}Minute_Bin": np.nan, f"{prefix}DayOfWeek": np.nan, f"{prefix}Month": np.nan, f"{prefix}DayOfYear": np.nan, f"{prefix}Is_Weekend": np.nan}
                        try:
                            dt = datetime.fromisoformat(iso_string)
                            return {
                                f"{prefix}Hour": dt.hour,
                                f"{prefix}Minute_Bin": dt.minute // 15, # Example binning
                                f"{prefix}DayOfWeek": dt.weekday(),
                                f"{prefix}Month": dt.month,
                                f"{prefix}DayOfYear": dt.dayofyear,
                                f"{prefix}Is_Weekend": int(dt.weekday() >= 5)
                            }
                        except ValueError:
                             return {f"{prefix}Hour": np.nan, f"{prefix}Minute_Bin": np.nan, f"{prefix}DayOfWeek": np.nan, f"{prefix}Month": np.nan, f"{prefix}DayOfYear": np.nan, f"{prefix}Is_Weekend": np.nan}

                    dt_features = pd.json_normalize(features_df[iso_col_name].apply(parse_dt))
                    features_df = pd.concat([features_df.drop(columns=[iso_col_name]), dt_features], axis=1)
                    print(f"Engineered datetime features with prefix '{prefix}' for {model_name}.")


        # Special handling for Passenger Volume Forecaster (Daily Total)
        if model_name == "passenger_volume_forecaster":
            if 'date_iso' in features_df.columns: # Input from UI for future dates
                temp_feature_list = []
                for idx, row in features_df.iterrows():
                    current_date = datetime.fromisoformat(row['date_iso']).date() # Date object
                    
                    # Engineer date parts
                    date_features = {
                        'Year': current_date.year, 'Month': current_date.month, 'Day': current_date.day,
                        'DayOfWeek': current_date.weekday(), 'DayOfYear': current_date.timetuple().tm_yday,
                        'WeekOfYear': current_date.isocalendar()[1], 'Quarter': (current_date.month - 1) // 3 + 1,
                        'Is_Weekend': int(current_date.weekday() >= 5),
                        'Time_Trend': (current_date - self.historical_daily_passengers_ts.index.min().date()).days + len(self.historical_daily_passengers_ts) if self.historical_daily_passengers_ts is not None and not self.historical_daily_passengers_ts.empty else np.nan # Approximation
                    }
                    
                    # Holiday feature
                    country_holidays = holidays.US(years=[current_date.year]) # Adjust country
                    date_features['Is_Holiday'] = int(current_date in country_holidays)

                    # Lag and Rolling Window Features (CRITICAL and COMPLEX for future dates)
                    # These need to be based on self.historical_daily_passengers_ts
                    # For future dates, you might only have lags up to the most recent historical data point.
                    # Or, if predicting multi-step ahead recursively, you'd use previous predictions.
                    # This example will use lags from the historical series.
                    if self.historical_daily_passengers_ts is not None:
                        lags_to_create = [1, 2, 3, 7, 14, 30]
                        for lag in lags_to_create:
                            lag_date = pd.Timestamp(current_date - pd.Timedelta(days=lag))
                            if lag_date in self.historical_daily_passengers_ts.index:
                                date_features[f'Lag_{lag}_Days'] = self.historical_daily_passengers_ts.loc[lag_date, 'TotalDailyPassengers']
                            else:
                                date_features[f'Lag_{lag}_Days'] = np.nan # Or use the most recent available lag

                        rolling_windows = [7, 14]
                        for window in rolling_windows:
                            # Rolling mean up to day before current_date - lag
                            end_date_for_roll = pd.Timestamp(current_date - pd.Timedelta(days=1))
                            start_date_for_roll = pd.Timestamp(end_date_for_roll - pd.Timedelta(days=window-1))
                            relevant_hist_data = self.historical_daily_passengers_ts.loc[
                                (self.historical_daily_passengers_ts.index >= start_date_for_roll) &
                                (self.historical_daily_passengers_ts.index <= end_date_for_roll),
                                'TotalDailyPassengers'
                            ]
                            if not relevant_hist_data.empty and len(relevant_hist_data) == window :
                                date_features[f'Rolling_Mean_{window}_Days'] = relevant_hist_data.mean()
                                date_features[f'Rolling_Std_{window}_Days'] = relevant_hist_data.std()
                            else:
                                date_features[f'Rolling_Mean_{window}_Days'] = np.nan
                                date_features[f'Rolling_Std_{window}_Days'] = np.nan
                    temp_feature_list.append(date_features)
                
                engineered_ts_features_df = pd.DataFrame(temp_feature_list, index=features_df.index)
                features_df = pd.concat([features_df.drop(columns=['date_iso']), engineered_ts_features_df], axis=1)
                print(f"Engineered time series features for {model_name}.")


        # --- Stage 2: Apply the fitted ColumnTransformer (preprocessor) ---
        # The preprocessor expects columns like 'Airline', 'Scheduled_Hour', etc.
        # It will handle imputation, scaling, and one-hot encoding internally as defined during training.
        try:
            print(f"DataHelper: Columns going INTO preprocessor for '{model_name}': {features_df.columns.tolist()}")
            # Ensure all columns expected by the preprocessor are present in features_df
            # The preprocessor.feature_names_in_ attribute (if scikit-learn >= 1.0) lists these.
            # If not available, this step relies on features_df having the right raw columns.
            if hasattr(preprocessor, 'feature_names_in_'):
                missing_cols_for_preprocessor = set(preprocessor.feature_names_in_) - set(features_df.columns)
                if missing_cols_for_preprocessor:
                    st.error(f"Missing columns required by the preprocessor for {model_name}: {missing_cols_for_preprocessor}")
                    for col in missing_cols_for_preprocessor: features_df[col] = np.nan # Add them as NaN so imputer can handle
            
            processed_data_np = preprocessor.transform(features_df) # This applies all steps in the ColumnTransformer

            # --- Stage 3: Reconstruct DataFrame with correct feature names AFTER transform ---
            if expected_feature_names_after_transform:
                if processed_data_np.shape[1] != len(expected_feature_names_after_transform):
                    st.error(f"Mismatch in column count for {model_name}! Processed data has {processed_data_np.shape[1]} cols, expected {len(expected_feature_names_after_transform)} based on saved feature names.")
                    st.error(f"Expected: {expected_feature_names_after_transform}")
                    # Try to use get_feature_names_out from preprocessor as a fallback
                    try:
                        actual_processed_names = list(preprocessor.get_feature_names_out())
                        st.warning(f"Using feature names from preprocessor.get_feature_names_out(): {actual_processed_names}")
                        if processed_data_np.shape[1] == len(actual_processed_names):
                             processed_features_df = pd.DataFrame(processed_data_np, columns=actual_processed_names, index=features_df.index)
                        else:
                            st.error("Shape mismatch even with get_feature_names_out. Preprocessing is likely misconfigured.")
                            return pd.DataFrame()
                    except Exception as e_getnames:
                        st.error(f"Could not get feature names from preprocessor for {model_name}: {e_getnames}")
                        return pd.DataFrame()
                else:
                     processed_features_df = pd.DataFrame(processed_data_np, columns=expected_feature_names_after_transform, index=features_df.index)
            else: # Fallback if feature_names_lists was not loaded (less robust)
                st.warning(f"Feature name list not available for {model_name}. Attempting to use preprocessor.get_feature_names_out().")
                try:
                    actual_processed_names = list(preprocessor.get_feature_names_out())
                    if processed_data_np.shape[1] == len(actual_processed_names):
                        processed_features_df = pd.DataFrame(processed_data_np, columns=actual_processed_names, index=features_df.index)
                    else:
                        st.error(f"Shape mismatch with get_feature_names_out for {model_name}. Expected {len(actual_processed_names)} got {processed_data_np.shape[1]}")
                        return pd.DataFrame()
                except Exception as e_getnames_fallback:
                    st.error(f"CRITICAL: Cannot determine processed feature names for {model_name}: {e_getnames_fallback}")
                    return pd.DataFrame()

            print(f"DataHelper: Preprocessing for '{model_name}' successful. Final columns: {processed_features_df.columns.tolist()}")
            return processed_features_df

        except Exception as e:
            st.error(f"Error during ColumnTransformer step for {model_name}: {e}")
            st.exception(e)
            return pd.DataFrame()

    # --- Your OTHER existing methods (get_data_summary, filter_data, etc.) ---
    # Remember to REVIEW AND UPDATE COLUMN NAMES in these methods to match your actual flights.csv
    # ... (paste your other methods here, then review them for column name consistency) ...
    def get_data_summary(self):
        # ... (your existing code) ...
        if self.data is not None:
            summary = {
                'total_records': len(self.data),
                'columns': list(self.data.columns),
                'date_range': None, # Initialize
                'airlines': None,   # Initialize
                'destinations': None # Initialize
            }
            if 'ScheduledTime' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['ScheduledTime']):
                summary['date_range'] = {
                    'start': self.data['ScheduledTime'].min(),
                    'end': self.data['ScheduledTime'].max()
                }
            if 'Airline' in self.data.columns:
                summary['airlines'] = sorted(self.data['Airline'].dropna().unique().tolist())
            if 'ArrivalAirport' in self.data.columns:
                summary['destinations'] = sorted(self.data['ArrivalAirport'].dropna().unique().tolist())
            return summary
        return None

    def filter_data(self, filters):
        # ... (your existing code, ensure column names match your CSV e.g., 'ScheduledTime', 'Airline', 'ArrivalAirport') ...
        if self.data is None:
            return None
        filtered_data = self.data.copy()
        if 'date_range' in filters and filters['date_range']:
            start_date, end_date = filters['date_range']
            if 'ScheduledTime' in filtered_data.columns and pd.api.types.is_datetime64_any_dtype(filtered_data['ScheduledTime']):
                filtered_data = filtered_data[
                    (filtered_data['ScheduledTime'].dt.date >= start_date) &
                    (filtered_data['ScheduledTime'].dt.date <= end_date)
                ]
        if 'airlines' in filters and filters['airlines']:
            if 'Airline' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Airline'].isin(filters['airlines'])]
        if 'destinations' in filters and filters['destinations']:
            if 'ArrivalAirport' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['ArrivalAirport'].isin(filters['destinations'])]
        return filtered_data

    def get_performance_metrics(self, data=None):
        if data is None: data = self.data
        if data is None or data.empty: return None
        metrics = {'total_flights': len(data)}
        if 'ArrivalDelay' in data.columns and pd.api.types.is_numeric_dtype(data['ArrivalDelay']):
            metrics['avg_arrival_delay'] = data['ArrivalDelay'].mean()
            metrics['ontime_arrival_rate'] = (data['ArrivalDelay'] <= 15).mean() * 100
            metrics['severe_arrival_delays'] = (data['ArrivalDelay'] >= 60).sum()
        if 'DepartureDelay' in data.columns and pd.api.types.is_numeric_dtype(data['DepartureDelay']):
            metrics['avg_departure_delay'] = data['DepartureDelay'].mean()
        if 'Passengers' in data.columns and pd.api.types.is_numeric_dtype(data['Passengers']):
            metrics['avg_passengers'] = data['Passengers'].mean()
            metrics['total_passengers'] = data['Passengers'].sum()
        return metrics

    def prepare_chart_data(self, chart_type, data=None):
        if data is None: data = self.data
        if data is None or data.empty: return None
        chart_data = {}
        if chart_type == 'hourly_volume':
            if 'ScheduledTime' in data.columns and pd.api.types.is_datetime64_any_dtype(data['ScheduledTime']):
                chart_data = data.groupby(data['ScheduledTime'].dt.hour).size().reset_index(name='count')
                chart_data.columns = ['hour', 'flights']
        elif chart_type == 'delay_distribution':
            if 'ArrivalDelay' in data.columns and pd.api.types.is_numeric_dtype(data['ArrivalDelay']):
                chart_data = data[['ArrivalDelay']].copy()
        elif chart_type == 'airline_performance':
            if 'Airline' in data.columns and 'ArrivalDelay' in data.columns:
                agg_dict = {'ArrivalDelay': ['mean', 'count']}
                if 'Passengers' in data.columns and pd.api.types.is_numeric_dtype(data['Passengers']):
                    agg_dict['Passengers'] = 'mean'
                chart_data = data.groupby('Airline').agg(agg_dict).round(2)
                chart_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in chart_data.columns.values]
                chart_data = chart_data.rename(columns={'ArrivalDelay_mean': 'avg_delay', 
                                                        'ArrivalDelay_count': 'flight_count',
                                                        'Passengers_mean': 'avg_passengers'}).reset_index()
        elif chart_type == 'gate_utilization':
            if 'Gate' in data.columns:
                chart_data = data.groupby('Gate').size().reset_index(name='usage_count')
        return chart_data

    def export_data(self, data, filename, format='csv'):
        # ... (your existing code) ...
        try:
            if format.lower() == 'csv': return data.to_csv(index=False)
            elif format.lower() == 'json': return data.to_json(orient='records', date_format='iso')
            return None
        except Exception as e: st.error(f"Error exporting data: {str(e)}"); return None

    def validate_data_quality(self):
        # ... (your existing code) ...
        if self.data is None: return ["No data loaded"]
        issues = []
        if self.data.isnull().sum().any(): issues.append(f"Missing values found in columns: {self.data.isnull().sum()[self.data.isnull().sum() > 0].index.tolist()}")
        if self.data.duplicated().sum() > 0: issues.append(f"Found {self.data.duplicated().sum()} duplicate records")
        if 'ArrivalDelay' in self.data.columns and not pd.api.types.is_numeric_dtype(self.data['ArrivalDelay']): issues.append("'ArrivalDelay' column is not numeric")
        if 'Passengers' in self.data.columns and not pd.api.types.is_numeric_dtype(self.data['Passengers']): issues.append("'Passengers' column is not numeric")
        # Add more checks as per your schema
        return issues if issues else ["Data quality looks good!"]