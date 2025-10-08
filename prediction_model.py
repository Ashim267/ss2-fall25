
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime


# Coordinates for Nashville, TN
LATITUDE = 36.17
LONGITUDE = -86.78


print("Fetching weather data from Open-Meteo...")

# The URL specifies coordinates, hourly data to get (temperature), and the unit.
url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&hourly=temperature_2m&temperature_unit=celsius&forecast_days=7"
)

response = requests.get(url)
data = response.json()

print("Successfully fetched forecast data.")

hourly_data = data['hourly']
df = pd.DataFrame(hourly_data)

# Convert the 'time' column from text to a real datetime format
df['time'] = pd.to_datetime(df['time'])

# Rename column for clarity
df = df.rename(columns={'temperature_2m': 'temperature_celsius'})

print("\n--- Data Head ---")
print(df.head())

# --- Visualize the Data ---
print("\nGenerating temperature forecast plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df['time'], df['temperature_celsius'], marker='.', linestyle='-', color='r')

ax.set_title(f'7-Day Temperature Forecast for Nashville', fontsize=16)
ax.set_xlabel('Date and Time', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Plot displayed.")

# --- Predictive Modeling ---
print("\nTraining a simple predictive model...")

# Prepare data: convert time to a numerical format (Unix timestamp)
df['timestamp'] = df['time'].apply(lambda x: x.timestamp())
X = df[['timestamp']]
y = df['temperature_celsius']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict the temperature 24 hours from the last forecast point
last_timestamp = df['timestamp'].iloc[-1]
future_timestamp = last_timestamp + (24 * 3600) # 24 hours in seconds

predicted_temp = model.predict([[future_timestamp]])
future_time = datetime.fromtimestamp(future_timestamp)

print("\n--- Prediction Result ---")
print(f"The model predicts the temperature at {future_time.strftime('%Y-%m-%d %H:%M')} will be: {predicted_temp[0]:.2f}°C")
