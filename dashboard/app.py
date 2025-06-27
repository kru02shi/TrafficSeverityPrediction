import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests

@st.cache_resource
def load_model():
#   url = "https://drive.google.com/file/d/1mfJf6Hqv9nmFBc1u_jrRh6HztmYPsECM/view?usp=sharing"
#   response = requests.get(url)
#   with open("model_rf.pkl", "wb") as f:
#       f.write(response.content)
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
    st.write(f"Loading mapping for: {col}")
    try:
        map_df = load_mapping(col)
        options = sorted(map_df[f"{col}_Original"].dropna().unique())
        st.write(f"Options for {col}: {options[:5]}...")  # show first few options
        selected = st.sidebar.selectbox(col, options)    
        code = map_df.loc[map_df[f"{col}_Original"] == selected, f"{col}_Code"].values[0]
        input_data[col] = code
    except FileNotFoundError:
        st.warning(f"Mapping file for {col} not found!")
        continue

x_input = pd.DataFrame([input_data]).reindex(columns=feature_names, fill_value=0)

# t1, t2, t3, t4, t5 = st.tabs(["Weather Alerts", "Planning", "EDA", "Predict Severity", "Traffic Control"])
t1, t2, t3, t4 = st.tabs(["Weather Alerts", "Planning", "EDA", "Predict Severity"])

with t1:
    st.header("Weather-Based Accident Alerts")
    st.write("Monitor risk based on current weather inputs.")
    if input_data["Weather_Condition"] != 0 or precip > 0:
        st.warning("High accident risk due to weather conditions!")
    else:
        st.success("Weather conditions are safe.")

with t2:
    st.header("Location-Based Planning")
    st.write("Identify locations needing maintenance based on high-severity accidents.")

    severity_counts = (
        df[df["Severity"] >= 3]  # Assuming Severity 3 & 4 are high
        .groupby("State")
        .size()
        .reset_index(name="High Severity Accidents")
        .sort_values("High Severity Accidents", ascending=False)
        .head(10)  # Top 10 cities with most high severity accidents
    )

    if not severity_counts.empty:
        bar_fig = px.bar(
            severity_counts,
            x="State",
            y="High Severity Accidents",
            labels={'State': 'State', 'High Severity Accidents': 'High Severity Accidents'},
            title="Top 10 Cities with High Severity Accidents"
        )
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.warning("No high-severity accidents found in the dataset.")

with t3:
    st.header("Exploratory Data Analysis")
    
    # 1) Severity Distribution
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Severity', data=df, palette='deep', ax=ax1)
    ax1.set_title('Accident Severity Distribution')
    ax1.set_xlabel('Severity Level')
    ax1.set_ylabel('Count')
    st.pyplot(fig1)

    # 2) Accidents by Hour of Day
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x='Hour', hue='Severity', multiple='stack',
                 bins=25, palette='mako', ax=ax2)
    ax2.set_title('Accidents by Hour of Day')
    ax2.set_xlabel('Hour (0–23)')
    ax2.set_ylabel('Count')
    st.pyplot(fig2)

    # 3) Accidents by Weekday
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Weekday', hue='Severity', data=df, palette='crest', ax=ax3)
    ax3.set_title('Accidents by Weekday')
    ax3.set_xlabel('Weekday')
    ax3.set_ylabel('Count')
    st.pyplot(fig3)

    # 4) Top 10 Weather Conditions
    top_weather = df['Weather_Condition'].value_counts().head(10).index
    weather_df = df[df['Weather_Condition'].isin(top_weather)]
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Weather_Condition', hue='Severity', data=weather_df,
                  order=top_weather, palette='Set2', ax=ax4)
    ax4.set_title('Accidents by Top 10 Weather Conditions')
    ax4.set_xlabel('Weather Condition')
    ax4.set_ylabel('Count')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig4)

    # 5) Top 10 States
    top_states = df['State'].value_counts().head(10).index
    state_df = df[df['State'].isin(top_states)]
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.countplot(x='State', hue='Severity', data=state_df,
                  order=top_states, palette='icefire', ax=ax5)
    ax5.set_title('Accidents by Top 10 States')
    ax5.set_xlabel('State')
    ax5.set_ylabel('Count')
    st.pyplot(fig5)

with t4:
    st.header("Severity Prediction")
    st.write("Predict accident severity based on input conditions.")
    if st.button("Predict Severity"):
        prediction = model.predict(x_input)[0]  # Adjust if your model shifted labels
        st.success(f"Predicted Severity Level: {prediction}")
