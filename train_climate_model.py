import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Replace with your data file, or create a simple dummy dataset for demo:
data = pd.DataFrame({
    "temperature": np.linspace(20, 30, 100),
    "co2": np.linspace(400, 450, 100),
    "rainfall": np.linspace(5, 25, 100)
})

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['temperature', 'co2', 'rainfall']])

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # temperature as target
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=5, batch_size=8)  # Set low epochs for quick testing

model.save("climate_lstm_model.h5")
joblib.dump(scaler, "climate_scaler.joblib")
print("Model and scaler saved.")
