from fastapi.middleware.cors import CORSMiddleware

# Add this after app = FastAPI()

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for development
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods
    allow_headers=["*"],  # allow all headers
)


model = tf.keras.models.load_model('climate_lstm_model.h5')
scaler = joblib.load('climate_scaler.joblib')

class WeatherInput(BaseModel):
    history: list  # List of [temp, co2, rainfall] for previous days

@app.post("/predict")
def predict(input: WeatherInput):
    X = np.array(input.history)[-10:]
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape((1, 10, 3))
    pred = model.predict(X_scaled)
    pred_temp = scaler.inverse_transform([[pred[0][0], 0, 0]])[0][0]
    return {"predicted_temperature": round(float(pred_temp), 2)}
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)