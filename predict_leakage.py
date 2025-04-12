# predict_leakage.py

import pandas as pd
import numpy as np
import os
from src.modeling.models import FreshLeakageModel, ReductionModel, CombinedModel

# Load the models
circuit_name = "NAND3"  # or your circuit name
models_dir = f"./models/{circuit_name}"

# Load fresh leakage model
fresh_model = FreshLeakageModel(model_dir=models_dir)
fresh_model.load(circuit_name=circuit_name)

# Load reduction models for different years
reduction_models = {}
years = [1, 2, 3, 5, 7, 10]  # Years of interest
seconds_per_year = 31536000.0

for year in years:
    reltime = year * seconds_per_year
    model = ReductionModel(model_dir=models_dir)
    
    try:
        model.load(circuit_name=circuit_name, reltime=reltime)
        reduction_models[reltime] = model
    except:
        print(f"No model found for year {year}")

# Create combined model
combined_model = CombinedModel(fresh_model, reduction_models)

# Define new circuit parameters
new_circuit = pd.DataFrame({
    'vin_a': [0.0],
    'vin_b': [0.0],
    'vin_c': [0.0],
    'tempu': [25.0],  # Example: 25°C
    'pvdd': [0.9],   # Example: 0.9V
    # Add all other required parameters...
})

# Predict leakage for different years
print(f"Predictions for circuit with temperature={new_circuit['tempu'][0]}°C and voltage={new_circuit['pvdd'][0]}V:")
print("-" * 80)

for year in years:
    if year * seconds_per_year in reduction_models:
        reltime = year * seconds_per_year
        prediction = combined_model.predict(new_circuit.values, reltime)
        
        print(f"Year {year}:")
        print(f"  - Fresh leakage: {prediction['fresh_leakage'][0][0]:.10e} A")
        print(f"  - Reduction: {prediction['reduction_percentage'][0][0]:.2f}%")
        print(f"  - Aged leakage: {prediction['aged_leakage'][0][0]:.10e} A")
        print("-" * 40)