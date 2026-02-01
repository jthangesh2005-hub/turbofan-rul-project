# Turbofan Engine RUL Prediction (FD001)

This project predicts the Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS FD001 dataset.

## Domain
Artificial Intelligence and Data Science (AIDS)

## Problem Statement
Predict the number of remaining operational cycles of a turbofan engine before failure based on sensor readings.

## Dataset
NASA CMAPSS Turbofan Engine Degradation Dataset (FD001)

## Approach
- Data cleaning and preprocessing
- RUL calculation and capping at 125 cycles
- Feature scaling using MinMaxScaler
- Model training using Gradient Boosting Regressor

## Model Performance
- Mean Absolute Error (MAE): ~13.5 cycles  
- Root Mean Squared Error (RMSE): ~18.7 cycles  

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Google Colab

## Outcome
The model accurately estimates engine Remaining Useful Life using sensor data, with low prediction error, making it suitable for predictive maintenance analysis.
