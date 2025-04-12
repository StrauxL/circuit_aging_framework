# setup.py

from setuptools import setup, find_packages

setup(
    name="circuit_aging_framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas==2.2.1",
        "scikit-learn==1.3.0",
        "catboost==1.2.3",
        "lightgbm==3.3.5",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pysr>=0.11.3",
        "pytest>=7.4.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "joblib>=1.2.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A framework for predicting leakage in aged circuits",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/circuit_aging_framework",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)