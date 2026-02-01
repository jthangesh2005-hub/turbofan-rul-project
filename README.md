# Turbofan Engine RUL Prediction (FD001)

This project predicts the Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS FD001 dataset.

## Domain
Artificial Intelligence and Data Science (AIDS)

## Problem Statement
To predict the number of remaining operational cycles of a turbofan engine before failure using sensor measurements.

## Dataset
NASA CMAPSS Turbofan Engine Degradation Dataset (FD001)

## Approach
- Data cleaning and preprocessing  
- RUL computation with capping at 125 cycles  
- Feature scaling using MinMaxScaler  
- Model training using Gradient Boosting Regressor  

## Model Performance
- **Mean Absolute Error (MAE):** ~13.5 cycles  
- **Root Mean Squared Error (RMSE):** ~18.7 cycles  

## Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  

## Outcome
The trained model effectively estimates the Remaining Useful Life of turbofan engines using sensor data, achieving low prediction error and demonstrating suitability for predictive maintenance applications.
