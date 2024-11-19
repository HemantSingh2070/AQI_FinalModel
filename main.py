import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("Air Quality Prediction App")


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
    "Vrindavan", "Yadgir", "Yamunanagar"
]

city = st.selectbox("Select a City", cities)

file_path = f"City_CSVs/{city}.csv"
try:
    air_quality_data = pd.read_csv(file_path)
    st.write(f"Data for {city} loaded successfully.")
    st.dataframe(air_quality_data)
except FileNotFoundError:
    st.error(f"No data file found for {city}. Please check the file path.")

if st.checkbox("Show Data Overview"):
    st.write("Head of the data:")
    st.dataframe(air_quality_data.head())
    st.write("Shape of the data:", air_quality_data.shape)
    st.write("Data Information:")
    st.text(air_quality_data.info())

st.write("Handling missing values...")
air_quality_data = air_quality_data.replace(to_replace=-200, value=np.nan)
air_quality_data.fillna(air_quality_data.mean(), inplace=True)

if st.checkbox("Show Missing Values Summary"):
    st.write("Missing values after cleaning:")
    st.write(air_quality_data.isnull().sum())

st.write("Processing dates...")
air_quality_data['Date'] = pd.to_datetime(air_quality_data['Date'], errors='coerce', dayfirst=True)
air_quality_data['time'] = "00:00:00"
air_quality_data['ds'] = air_quality_data['Date'].astype(str) + " " + air_quality_data['time']
data = pd.DataFrame()
data['ds'] = pd.to_datetime(air_quality_data['ds'])

pollutants = air_quality_data.columns[1:]
y = st.selectbox("Select a Pollutant", pollutants)

data['y'] = air_quality_data[y]

freq = st.selectbox("Select Prediction Frequency", ["D (Daily)", "W (Weekly)", "M (Monthly)"], index=2)

freq_code = freq.split(" ")[0]

st.write("Building the Prophet model...")
model = Prophet()
model.fit(data)

st.write("Generating future predictions...")
future = model.make_future_dataframe(periods=30, freq=freq_code)
forecast = model.predict(future)

if st.checkbox("Show Forecast Data"):
    st.write(forecast.tail())

st.write("Generating plots...")
st.write("Forecast Plot:")
forecast_fig = model.plot(forecast)
st.pyplot(forecast_fig)

st.write("Components Plot:")
components_fig = model.plot_components(forecast)
st.pyplot(components_fig)
