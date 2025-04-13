# Circuit Aging Leakage Prediction Framework

A comprehensive framework for predicting leakage in aged electronic circuits by combining fresh simulation predictions with aging-induced reduction percentages. This framework automates the training of machine learning models, pattern analysis, and visualization for circuit aging research.

## Features

- Automated data processing for fresh and aged circuit datasets
- Machine learning model training for fresh leakage and reduction percentage prediction
- Combined prediction of aged leakage values
- Pattern analysis and visualization of leakage reduction trends
- Symbolic regression to find mathematical formulas for aging patterns
- Support for multiple circuit types (NAND3, XOR2, etc.)
- Comprehensive documentation and visualization for research papers

## Project Structure
circuit_aging_framework/
├── data/                    # Store all dataset files
├── models/                  # Saved trained models
├── results/                 # Results, visualizations, and analysis
├── src/                     # Source code
│   ├── data_processing/         # Data processing utilities
│   ├── modeling/                # ML model implementations
│   ├── analysis/                # Analysis and visualization tools
│   ├── symbolic_regression/     # PySR implementation
│   └── utils/                   # Utility functions
├── notebooks/               # Jupyter notebooks for experiments
├── config/                  # Configuration files
├── requirements.txt         # Project dependencies
├── setup.py                 # Package setup script
└── README.md                # Project documentation


## Installation Guide

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/circuit_aging_framework.git
cd circuit_aging_framework
#python -m venv venv
#source venv/bin/activate
conda create --new myenv python=3.11.12
pip install -r requirements.txt
```
### Step 2: Add NAND3_Cleaned.csv and NAND3_Final_Leakage_Reduction.csv in data folder
### Step 3: python process_new_circuit.py --fresh data/NAND3_Cleaned.csv --reduction data/NAND3_Final_Leakage_Reduction.csv --circuit NAND3
