# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from scipy.spatial.distance import cdist

# Define past data
past_data = {
    'pincode': [411028, 411028, 411037, 411008, 411038, 411058, 411004, 411004],
    'latitude': [18.5004347, 18.4752589, 18.4672524, 18.54072, 18.4467405, 18.4804818, 18.5144406, 18.5219448],
    'longitude': [73.9395879, 73.924989, 73.8641823, 73.822497, 73.8585979, 73.8009341, 73.8332788, 73.8475182],
    'time_period': ['morning', 'afternoon', 'evening', 'night', 'morning', 'afternoon', 'evening', 'night'],
    'crime_type': ['burglary', 'assault', 'theft', 'robbery', 'burglary', 'assault', 'theft', 'robbery'],
    'date': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-02', '2024-01-02'],
}

# Create DataFrame for past data
past_df = pd.DataFrame(past_data)

# Preprocess the data
X = past_df[['longitude', 'latitude', 'pincode']].values
y_lon = past_df['longitude'].values
y_lat = past_df['latitude'].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_lon_scaled = scaler_y.fit_transform(y_lon.reshape(-1, 1))
y_lat_scaled = scaler_y.fit_transform(y_lat.reshape(-1, 1))

X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Split the data into training and testing sets
X_train, X_test, y_lon_train, y_lon_test, y_lat_train, y_lat_test = train_test_split(X_lstm, y_lon_scaled, y_lat_scaled, test_size=0.2, random_state=42)

# Define the LSTM model for longitude prediction
model_lon = Sequential()
model_lon.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model_lon.add(LSTM(units=50))
model_lon.add(Dense(units=1, activation='linear'))  # Linear activation for regression
model_lon.compile(optimizer='adam', loss='mean_squared_error')
model_lon.fit(X_train, y_lon_train, epochs=50, batch_size=32, validation_data=(X_test, y_lon_test))

# Define the LSTM model for latitude prediction
model_lat = Sequential()
model_lat.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model_lat.add(LSTM(units=50))
model_lat.add(Dense(units=1, activation='linear'))  # Linear activation for regression
model_lat.compile(optimizer='adam', loss='mean_squared_error')
model_lat.fit(X_train, y_lat_train, epochs=50, batch_size=32, validation_data=(X_test, y_lat_test))

# Prepare input conditions for prediction (longitude, latitude, pincode)
input_conditions = np.array([[0.5, 0.5, 0.5]])  # Adjust values accordingly

# Normalize the input conditions using the same scaler used for training
input_conditions_scaled = scaler_X.transform(input_conditions)

# Reshape for LSTM input
input_conditions_lstm = input_conditions_scaled.reshape(1, 1, input_conditions_scaled.shape[1])

# Predict future longitude and latitude coordinates
future_lon_scaled = model_lon.predict(input_conditions_lstm)
future_lat_scaled = model_lat.predict(input_conditions_lstm)

# Inverse transform the scaled predictions to get actual longitude and latitude
future_lon = scaler_y.inverse_transform(future_lon_scaled)
future_lat = scaler_y.inverse_transform(future_lat_scaled)

# Find the nearest pincode to the predicted future longitude and latitude
future_coordinates = np.array([future_lon, future_lat]).reshape(1, -1)
past_coordinates = past_df[['longitude', 'latitude']].values
distances = cdist(future_coordinates, past_coordinates, 'euclidean')
nearest_index = np.argmin(distances)
nearest_pincode = past_df.iloc[nearest_index]['pincode']

# Print predicted future longitude, latitude, and nearest pincode
print("Predicted future longitude:", future_lon)
print("Predicted future latitude:", future_lat)
print("Nearest pincode:", nearest_pincode)




# Make predictions on the test set for both longitude and latitude
predicted_lon_scaled = model_lon.predict(X_test)
predicted_lat_scaled = model_lat.predict(X_test)

# Inverse transform the scaled predictions to get actual longitude and latitude
predicted_lon = scaler_y.inverse_transform(predicted_lon_scaled)
predicted_lat = scaler_y.inverse_transform(predicted_lat_scaled)

# Calculate evaluation metrics (e.g., MSE, MAE, RMSE)
from sklearn.metrics import mean_squared_error, mean_absolute_error

lon_mse = mean_squared_error(y_lon_test, predicted_lon)
lat_mse = mean_squared_error(y_lat_test, predicted_lat)

lon_mae = mean_absolute_error(y_lon_test, predicted_lon)
lat_mae = mean_absolute_error(y_lat_test, predicted_lat)

print("Longitude MSE:", lon_mse)
print("Latitude MSE:", lat_mse)
print("Longitude MAE:", lon_mae)
print("Latitude MAE:", lat_mae)
