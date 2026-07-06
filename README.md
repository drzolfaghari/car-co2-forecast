# CO2 Emission Prediction using Linear Regression

This project predicts vehicle CO2 emissions (g/km) using Linear Regression based on engine size, cylinder count, and fuel consumption.

## Dataset
- **Records**: 500 vehicles
- **Features**: engine (L), cylandr (cylinders), fuelcomb (L/100km)
- **Target**: out1 (CO2 emissions in g/km)

## Model Performance
- **R² Score**: 0.8531
- **MAE**: 22.40
- **RMSE**: 29.64

## Installation
```bash
pip install -r requirements.txt

## Usage
```python
import joblib
import numpy as np

model = joblib.load('co2_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_co2(engine, cylinders, fuel_comb):
    data = scaler.transform([[engine, cylinders, fuel_comb]])
    return model.predict(data)[0]

print(predict_co2(2.5, 4, 9.0))