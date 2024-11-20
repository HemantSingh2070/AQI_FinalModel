import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Application Title
st.title("Air Quality Prediction App")

# List of cities for selection
cities = [
    "Agartala", "Agra", "Ahmedabad", "Aizawl", "Ajmer", "Akola", "Alwar", "Amaravati", "Ambala",
    "Amravati", "Amritsar", "Anantapur", "Angul", "Ankleshwar", "Araria", "Ariyalur", "Arrah",
    "Asansol", "Aurangabad", "Aurangabad (Bihar)", "Baddi", "Badlapur", "Bagalkot", "Baghpat",
    "Bahadurgarh", "Balasore", "Ballabgarh", "Banswara", "Baran", "Barbil", "Bareilly", "Baripada",
    "Barmer", "Barrackpore", "Bathinda", "Begusarai", "Belapur", "Belgaum", "Bengaluru", "Bettiah",
    "Bhagalpur", "Bharatpur", "Bhilai", "Bhilwara", "Bhiwadi", "Bhiwandi", "Bhiwani", "Bhopal",
    "Bhubaneswar", "Bidar", "Bihar Sharif", "Bikaner", "Bilaspur", "Bileipada", "Brajrajnagar",
    "Bulandshahr", "Bundi", "Buxar", "Byasanagar", "Byrnihat", "Chamarajanagar", "Chandigarh",
    "Chandrapur", "Charkhi Dadri", "Chengalpattu", "Chennai", "Chhal", "Chhapra", "Chikkaballapur",
    "Chikkamagaluru", "Chittoor", "Chittorgarh", "Churu", "Coimbatore", "Cuddalore", "Cuttack",
    "Damoh", "Darbhanga", "Dausa", "Davanagere", "Dehradun", "Delhi", "Dewas", "Dhanbad", "Dharuhera",
    "Dharwad", "Dholpur", "Dhule", "Dindigul", "Durgapur", "Eloor", "Ernakulam", "Faridabad", "Fatehabad",
    "Firozabad", "Gadag", "GandhiNagar", "Gangtok", "Gaya", "Ghaziabad", "Gorakhpur", "Greater Noida",
    "Gummidipoondi", "Gurugram", "Guwahati", "Gwalior", "Hajipur", "Haldia", "Hanumangarh", "Hapur",
    "Hassan", "Haveri", "Hisar", "Hosur", "Howrah", "Hubballi", "Hyderabad", "Imphal", "Indore",
    "Jabalpur", "Jaipur", "Jaisalmer", "Jalandhar", "Jalgaon", "Jalna", "Jalore", "Jhalawar", "Jhansi",
    "Jharsuguda", "Jhunjhunu", "Jind", "Jodhpur", "Jorapokhar", "Kadapa", "Kaithal", "Kalaburagi",
    "Kalyan", "Kanchipuram", "Kannur", "Kanpur", "Karauli", "Karnal", "Karwar", "Kashipur", "Katihar",
    "Katni", "Keonjhar", "Khanna", "Khurja", "Kishanganj", "Kochi", "Kohima", "Kolar", "Kolhapur",
    "Kolkata", "Kollam", "Koppal", "Korba", "Kota", "Kozhikode", "Kunjemura", "Kurukshetra", "Latur",
    "Loni_Dehat", "Loni_Ghaziabad", "Lucknow", "Ludhiana", "Madikeri", "Mahad", "Maihar", "Mandi Gobindgarh",
    "Mandideep", "Mandikhera", "Manesar", "Mangalore", "Manguraha", "Medikeri", "Meerut", "Milupara",
    "Moradabad", "Motihari", "Mumbai", "Munger", "Muzaffarnagar", "Muzaffarpur", "Mysuru", "Nagaon",
    "Nagaur", "Nagpur", "Naharlagun", "Nalbari", "Nanded", "Nandesari", "Narnaul", "Nashik", "Navi Mumbai",
    "Nayagarh", "Noida", "Ooty", "Pali", "Palkalaiperur", "Palwal", "Panchkula", "Panipat", "Parbhani",
    "Patiala", "Patna", "Pimpri Chinchwad", "Pithampur", "Pratapgarh", "Prayagraj", "Puducherry", "Pune",
    "Purnia", "Raichur", "Raipur", "Rairangpur", "Rajamahendravaram", "Rajgir", "Rajsamand", "Ramanagara",
    "Ramanathapuram", "Ratlam", "Rishikesh", "Rohtak", "Rourkela", "Rupnagar", "Sagar", "Saharsa", "Salem",
    "Samastipur", "Sangli", "Sasaram", "Satna", "Sawai Madhopur", "Shillong", "Shivamogga", "Sikar", "Silchar",
    "Siliguri", "Singrauli", "Sirohi", "Sirsa", "Sivasagar", "Siwan", "Solapur", "Sonipat", "Sri Ganganagar",
    "Srinagar", "Suakati", "Surat", "Talcher", "Tensa", "Thane", "Thiruvananthapuram", "Thoothukudi", "Thrissur",
    "Tiruchirappalli", "Tirupati", "Tirupur", "Tonk", "Tumakuru", "Tumidih", "Udaipur", "Udupi", "Ujjain",
    "Ulhasnagar", "Vapi", "Varanasi", "Vatva", "Vellore", "Vijayapura", "Vijayawada", "Visakhapatnam",
    "Vrindavan", "Yadgir", "Yamunanagar"]

# Select a city from the dropdown
city = st.selectbox("Select a City", cities)

# Construct the file path dynamically
file_path = f"{city}.csv"

# Load city-specific air quality data
try:
    air_quality_data = pd.read_csv(file_path)
    st.success(f"Data for {city} loaded successfully.")
    st.dataframe(air_quality_data)
except FileNotFoundError:
    st.error(f"No data file found for {city}. Please ensure the file path is correct.")
    st.stop()  # Stop execution if file is not found

# Show basic data overview
if st.checkbox("Show Data Overview"):
    st.write("Data Preview:")
    st.dataframe(air_quality_data.head())
    st.write(f"Shape of the Data: {air_quality_data.shape}")
    st.write("Data Information:")
    st.text(air_quality_data.info())

# Handle missing values
st.write("Cleaning missing values...")
air_quality_data.replace(to_replace=-200, value=np.nan, inplace=True)
air_quality_data.fillna(air_quality_data.mean(), inplace=True)

# Summary of missing values
if st.checkbox("Show Missing Values Summary"):
    st.write("Missing values after cleaning:")
    st.write(air_quality_data.isnull().sum())

# Process dates for Prophet model
st.write("Processing date columns...")
try:
    air_quality_data['Date'] = pd.to_datetime(air_quality_data['Date'], errors='coerce', dayfirst=True)
    air_quality_data['time'] = "00:00:00"
    air_quality_data['ds'] = air_quality_data['Date'].astype(str) + " " + air_quality_data['time']
    data = pd.DataFrame()
    data['ds'] = pd.to_datetime(air_quality_data['ds'])
except KeyError:
    st.error("Date column is missing or incorrectly formatted in the dataset.")
    st.stop()

# Select pollutant for prediction
pollutants = list(air_quality_data.columns[1:])  # Exclude date column
if not pollutants:
    st.error("No pollutants found in the dataset.")
    st.stop()

y = st.selectbox("Select a Pollutant for Prediction", pollutants)

# Prepare data for Prophet
data['y'] = air_quality_data[y]

# Select prediction frequency
freq = st.selectbox("Select Prediction Frequency", ["D (Daily)", "W (Weekly)", "M (Monthly)"], index=2)
freq_code = freq.split(" ")[0]  # Extract frequency code

# Train the Prophet model
st.write("Building the Prophet model...")
try:
    model = Prophet()
    model.fit(data)
except Exception as e:
    st.error(f"Error training the model: {e}")
    st.stop()

# Generate future predictions
st.write("Generating future predictions...")
future = model.make_future_dataframe(periods=30, freq=freq_code)
forecast = model.predict(future)

# Display forecast data
if st.checkbox("Show Forecast Data"):
    st.write(forecast.tail())

# Generate and display plots
st.write("Generating Plots...")

# Forecast Plot
try:
    st.write("Forecast Plot:")
    forecast_fig = model.plot(forecast)
    st.pyplot(forecast_fig)
except Exception as e:
    st.error(f"Error generating forecast plot: {e}")

# Components Plot
try:
    st.write("Components Plot:")
    components_fig = model.plot_components(forecast)
    st.pyplot(components_fig)
except Exception as e:
    st.error(f"Error generating components plot: {e}")
