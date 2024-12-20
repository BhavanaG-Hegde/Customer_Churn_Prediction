import joblib
import numpy as np

# Load the model and scaler
lr = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# Sample input (age, balance, active, num_products, gender_male)
sample_input = [[30, 100000, 0, 1, 0]]  # Example input, adjust based on your data

# Scale the input data
input_scaled = scaler.transform(sample_input)

# Predict churn probability
res = lr.predict_proba(input_scaled)[:, 0]

print("Churn probability:", res[0] * 100)
