import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("../models/model_rf.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("../data/accidents_cleaned.csv")
    X = pd.read_csv("../data/input_cols.csv")       # encoded data for prediction task
    return df, X

@st.cache_data
def load_mapping(col):
    return pd.read_csv(f"../data/{col}_encoding.csv")

model = load_model()
df, X = load_data()
feature_names = X.columns

categorical_cols = X.select_dtypes(include='object').columns.tolist()

st.set_page_config(page_title="Traffic Accident Severity Dashboard", layout="wide")

st.title("Traffic Accident Severity Predictor & Dashboard")

# Sidebar for input features
st.sidebar.header("Predict Accident Severity")

start_lat = st.sidebar.number_input("Start Latitude", value=37.77)
start_lng = st.sidebar.number_input("Start Longitude", value=-122.42)
distance = st.sidebar.slider("Distance (mi)", 0.0, 100.0, 1.0, step=0.1)
temp = st.sidebar.slider("Temperature (F)", -20, 120, 70)
wind_chill = st.sidebar.slider("Wind Chill (F)", -30, 100, 60)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
pressure = st.sidebar.slider("Pressure (in)", 20.0, 40.0, 30.0, step=0.1)
visibility = st.sidebar.slider("Visibility (mi)", 0.0, 10.0, 5.0, step=0.1)
wind_speed = st.sidebar.slider("Wind Speed (mph)", 0.0, 50.0, 10.0, step=0.5)
precip = st.sidebar.slider("Precipitation (in)", 0.0, 5.0, 0.0, step=0.1)
hour = st.sidebar.slider("Hour", 0, 23, 12)
day = st.sidebar.slider("Day", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)
weekday = st.sidebar.slider("Weekday", 0, 6, 2)

# Convert to model input (adjust as per your model)
input_data = {col: 0 for col in feature_names}

input_data.update({
    'Start_Lat': start_lat,
    'Start_Lng': start_lng,
    'Distance(mi)': distance,
    'Temperature(F)': temp,
    'Wind_Chill(F)': wind_chill,
    'Humidity(%)': humidity,
    'Pressure(in)': pressure,
    'Visibility(mi)': visibility,
    'Wind_Speed(mph)': wind_speed,
    'Precipitation(in)': precip,
    'Hour': hour,
    'Day': day,
    'Month': month,
    'Weekday': weekday
})

for col in categorical_cols:
    map_df = load_mapping(col)
    options = sorted(map_df[f"{col}_Original"].dropna().unique())
    selected = st.sidebar.selectbox(col, options)    
    code = map_df.loc[map_df[f"{col}_Original"] == selected, f"{col}_Code"].values[0]
    input_data[col] = code

x_input = pd.DataFrame([input_data]).reindex(columns=feature_names, fill_value=0)

if st.sidebar.button("Predict Severity"):
    prediction = model.predict(x_input)[0]  # Adjust if your model shifted labels
    st.sidebar.success(f"Predicted Severity Level: {prediction}")

st.sidebar.header("Hotspot Map Filters")
lat_min = st.sidebar.number_input("Min Latitude", value=float(df["Start_Lat"].min()))
lat_max = st.sidebar.number_input("Max Latitude", value=float(df["Start_Lat"].max()))
lng_min = st.sidebar.number_input("Min Longitude", value=float(df["Start_Lng"].min()))
lng_max = st.sidebar.number_input("Max Longitude", value=float(df["Start_Lng"].max()))

df_filtered = df[
    (df["Start_Lat"] >= lat_min) & (df["Start_Lat"] <= lat_max) &
    (df["Start_Lng"] >= lng_min) & (df["Start_Lng"] <= lng_max)
]

# Tabs for use cases
t1, t2, t3 = st.tabs(["Traffic Control", "Weather Alerts", "Planning"])

with t1:
    st.header("Accident Hotspots & Traffic Control")
    st.write("Visualize accident-prone locations for better traffic management.")
    if not df_filtered.empty:
        fig = px.density_mapbox(
            df_filtered,
            lat="Start_Lat",
            lon="Start_Lng",
            z="Severity",  # or use count / weight
            radius=10,  # adjust for density granularity
            center=dict(lat=df_filtered["Start_Lat"].mean(), lon=df_filtered["Start_Lng"].mean()),
            zoom=3,
            mapbox_style="carto-positron"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data points in the selected coordinate range.")

with t2:
    st.header("Weather-Based Accident Alerts")
    st.write("Monitor risk based on current weather inputs.")
    if input_data["Weather_Condition"] != 0 or precip > 0:
        st.warning("High accident risk due to weather conditions!")
    else:
        st.success("Weather conditions are safe.")

with t3:
    st.header("Location-Based Planning")
    st.write("Identify locations needing maintenance.")
    bar_fig = px.bar(x=["City A", "City B", "City C"], y=[120, 80, 60], labels={'x': 'City', 'y': 'High Severity Accidents'})
    st.plotly_chart(bar_fig, use_container_width=True)
