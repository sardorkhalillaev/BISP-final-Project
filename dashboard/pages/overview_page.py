import streamlit as st
import pandas as pd
import plotly.express as px

def render_page(df, data_helper): # model_helper not needed here
    st.header("Airport Performance Overview")

    if df is None or df.empty:
        st.warning("No data loaded. Please load data from the sidebar to view the overview.")
        return

    # --- KPIs ---
    st.subheader("Key Performance Indicators")
    total_flights = len(df)
    
    # Calculate Avg Arrival Delay (ensure ARRIVAL_DELAY is numeric)
    avg_arrival_delay_str = "N/A"
    if 'ArrivalDelay' in df.columns and pd.to_numeric(df['ArrivalDelay'], errors='coerce').notnull().any():
        avg_arrival_delay = pd.to_numeric(df['ArrivalDelay'], errors='coerce').mean()
        avg_arrival_delay_str = f"{avg_arrival_delay:.2f} min"

    # Calculate Avg Departure Delay
    avg_departure_delay_str = "N/A"
    if 'DepartureDelay' in df.columns and pd.to_numeric(df['DepartureDelay'], errors='coerce').notnull().any():
        avg_departure_delay = pd.to_numeric(df['DepartureDelay'], errors='coerce').mean()
        avg_departure_delay_str = f"{avg_departure_delay:.2f} min"

    # On-Time Performance (Example: ArrivalDelay <= 15 minutes)
    on_time_percentage_str = "N/A"
    if 'ArrivalDelay' in df.columns and pd.to_numeric(df['ArrivalDelay'], errors='coerce').notnull().any():
        on_time_flights = df[pd.to_numeric(df['ArrivalDelay'], errors='coerce') <= 15].shape[0]
        if total_flights > 0:
            on_time_percentage = (on_time_flights / total_flights) * 100
            on_time_percentage_str = f"{on_time_percentage:.2f}%"
    
    # Total Passengers (if available directly, or sum if model gives per flight)
    total_passengers_str = "N/A"
    if 'Passengers' in df.columns and pd.to_numeric(df['Passengers'], errors='coerce').notnull().any():
        total_passengers = pd.to_numeric(df['Passengers'], errors='coerce').sum()
        total_passengers_str = f"{total_passengers:,.0f}"


    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Total Flights", str(total_flights))
    with kpi_cols[1]:
        st.metric("Avg. Arrival Delay", avg_arrival_delay_str)
    with kpi_cols[2]:
        st.metric("Avg. Departure Delay", avg_departure_delay_str)
    with kpi_cols[3]:
        st.metric("On-Time Arrival %", on_time_percentage_str, help="Flights with Arrival Delay <= 15 min")
    
    # --- Visualizations ---
    st.markdown("---")
    st.subheader("Visualizations")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.write("##### Flights by Airline (Top 10)")
        if 'Airline' in df.columns:
            airline_counts = df['Airline'].value_counts().nlargest(10)
            if not airline_counts.empty:
                fig = px.bar(airline_counts, x=airline_counts.index, y=airline_counts.values,
                             labels={'x':'Airline', 'y':'Number of Flights'}, color=airline_counts.index)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No airline data to display.")
        else:
            st.warning("Column 'Airline' not found.")

        st.write("##### Delay Distribution (Arrival)")
        if 'ArrivalDelay' in df.columns and pd.to_numeric(df['ArrivalDelay'], errors='coerce').notnull().any():
            delay_data = pd.to_numeric(df['ArrivalDelay'], errors='coerce').dropna()
            delay_data_filtered = delay_data[(delay_data > -60) & (delay_data < 180)] # Filter outliers
            if not delay_data_filtered.empty:
                fig_hist = px.histogram(delay_data_filtered, nbins=50, labels={'value':'Arrival Delay (minutes)'})
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No arrival delay data in the typical range.")
        else:
            st.warning("Column 'ArrivalDelay' not found or non-numeric.")

    with viz_col2:
        st.write("##### Flights by Weather Condition")
        if 'WeatherCondition' in df.columns:
            weather_counts = df['WeatherCondition'].value_counts().nlargest(10)
            if not weather_counts.empty:
                fig_pie = px.pie(weather_counts, names=weather_counts.index, values=weather_counts.values,
                                 title="Flights by Weather Condition (Top 10)")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No weather condition data.")
        else:
            st.warning("Column 'WeatherCondition' not found.")
        
        st.write("##### Delay Distribution (Departure)")
        if 'DepartureDelay' in df.columns and pd.to_numeric(df['DepartureDelay'], errors='coerce').notnull().any():
            delay_data_dep = pd.to_numeric(df['DepartureDelay'], errors='coerce').dropna()
            delay_data_dep_filtered = delay_data_dep[(delay_data_dep > -60) & (delay_data_dep < 180)]
            if not delay_data_dep_filtered.empty:
                fig_hist_dep = px.histogram(delay_data_dep_filtered, nbins=50, labels={'value':'Departure Delay (minutes)'})
                st.plotly_chart(fig_hist_dep, use_container_width=True)
            else:
                st.info("No departure delay data in the typical range.")
        else:
            st.warning("Column 'DepartureDelay' not found or non-numeric.")
            
    # Add more visualizations as needed, e.g., flights over time, delays by airport, etc.