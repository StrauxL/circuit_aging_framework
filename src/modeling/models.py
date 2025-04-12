# src/modeling/models.py

import os
import numpy as np
import pandas as pd
import logging
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreshLeakageModel:
    """Model for predicting leakage in fresh circuit simulations."""
    
    def __init__(self, model_dir='./models'):
        """
        Initialize the model.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save model files
        """
        self.model_dir = model_dir
        self.model = None
        os.makedirs(model_dir, exist_ok=True)
    
    def train(self, prepared_data, model_params=None):
        """
        Train the model.
        
        Parameters:
        -----------
        prepared_data : dict
            Dictionary containing prepared data from CircuitDataLoader
        model_params : dict, optional
            Parameters for the CatBoost model
            
        Returns:
        --------
        self
        """
        # Extract data
        X_train_scaled = prepared_data['X_train_scaled']
        y_train_scaled = prepared_data['y_train_scaled']
        
        # Set default model parameters if not provided
        if model_params is None:
            model_params = {
                'iterations': 1000,
                'depth': 16,
                'learning_rate': 0.05,
                'loss_function': 'RMSE',
                'random_seed': 1,
                'verbose': 0,
                'task_type': "GPU"  # Change to "GPU" if available
            }
        
        logger.info("Training fresh leakage model...")
        
        # Initialize and train the model
        self.model = CatBoostRegressor(**model_params)
        self.model.fit(X_train_scaled, y_train_scaled.ravel())
        
        logger.info("Fresh leakage model training completed")
        return self
    
    def evaluate(self, prepared_data):
        """
        Evaluate the model.
        
        Parameters:
        -----------
        prepared_data : dict
            Dictionary containing prepared data from CircuitDataLoader
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")
        
        # Extract data
        X_train_scaled = prepared_data['X_train_scaled']
        X_test_scaled = prepared_data['X_test_scaled']
        y_train_scaled = prepared_data['y_train_scaled']
        y_test = prepared_data['y_test']
        scaler_output = prepared_data['scaler_output']
        
        # Make predictions
        y_train_pred_scaled = self.model.predict(X_train_scaled)
        y_train_pred = scaler_output.inverse_transform(y_train_pred_scaled.reshape(-1, 1))
        
        y_test_pred_scaled = self.model.predict(X_test_scaled)
        y_test_pred = scaler_output.inverse_transform(y_test_pred_scaled.reshape(-1, 1))
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train_scaled, y_train_pred_scaled)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        train_mae = mean_absolute_error(y_train_scaled, y_train_pred_scaled)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        train_mape = mean_absolute_percentage_error(y_train_scaled, y_train_pred_scaled) * 100
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
        
        train_r2 = r2_score(y_train_scaled, y_train_pred_scaled)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Log metrics
        logger.info(f"Train MSE: {train_mse}")
        logger.info(f"Test MSE: {test_mse}")
        logger.info(f"Train MAE: {train_mae}")
        logger.info(f"Test MAE: {test_mae}")
        logger.info(f"Train MAPE: {train_mape}%")
        logger.info(f"Test MAPE: {test_mape}%")
        logger.info(f"Train R²: {train_r2}")
        logger.info(f"Test R²: {test_r2}")
        
        # Return metrics dictionary
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'y_test': y_test,  # Add the actual y_test values
            'y_train': scaler_output.inverse_transform(y_train_scaled)  # Add the actual y_train values
        }
    
    def predict(self, X, scaler_input=None, scaler_output=None):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : array-like
            Input features
        scaler_input : sklearn.preprocessing.StandardScaler, optional
            Scaler used to transform input features
        scaler_output : sklearn.preprocessing.StandardScaler, optional
            Scaler used to inverse transform predictions
            
        Returns:
        --------
        array
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")
        
        # Scale input if scaler provided
        if scaler_input is not None:
            X_scaled = scaler_input.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Inverse transform if scaler provided
        if scaler_output is not None:
            y_pred = scaler_output.inverse_transform(y_pred_scaled.reshape(-1, 1))
        else:
            y_pred = y_pred_scaled
        
        return y_pred
    
    def save(self, circuit_name="default"):
        """
        Save the model.
        
        Parameters:
        -----------
        circuit_name : str
            Name of the circuit for the model filename
            
        Returns:
        --------
        str
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")
        
        model_path = os.path.join(self.model_dir, f"fresh_leakage_{circuit_name}.cbm")
        self.model.save_model(model_path)
        logger.info(f"Fresh leakage model saved to {model_path}")
        return model_path
    
    def load(self, model_path=None, circuit_name="default"):
        """
        Load a saved model.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to the saved model file
        circuit_name : str, optional
            Name of the circuit for the model filename
            
        Returns:
        --------
        self
        """
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"fresh_leakage_{circuit_name}.cbm")
        
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)
        logger.info(f"Fresh leakage model loaded from {model_path}")
        return self


class ReductionModel:
    """Model for predicting leakage reduction percentages."""
    
    def __init__(self, model_dir='./models'):
        """
        Initialize the model.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save model files
        """
        self.model_dir = model_dir
        self.model = None
        self.reltime = None
        os.makedirs(model_dir, exist_ok=True)
    
    def train(self, prepared_data, model_params=None):
        """
        Train the model.
        
        Parameters:
        -----------
        prepared_data : dict
            Dictionary containing prepared data from CircuitDataLoader
        model_params : dict, optional
            Parameters for the LightGBM model
            
        Returns:
        --------
        self
        """
        # Extract data
        X_train_scaled = prepared_data['X_train_scaled']
        y_train_scaled = prepared_data['y_train_scaled']
        self.reltime = prepared_data['reltime']
        
        # Set default model parameters if not provided
        if model_params is None:
            model_params = {
                'n_estimators': 1000,
                'n_jobs': -1,
                'num_leaves': 500
            }
        
        logger.info(f"Training reduction model for reltime {self.reltime}...")
        
        # Initialize and train the model
        self.model = lgb.LGBMRegressor(**model_params)
        self.model.fit(X_train_scaled, y_train_scaled.ravel())
        
        logger.info(f"Reduction model training completed for reltime {self.reltime}")
        return self
    
    def evaluate(self, prepared_data):
        """
        Evaluate the model.
        
        Parameters:
        -----------
        prepared_data : dict
            Dictionary containing prepared data from CircuitDataLoader
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")
        
        # Extract data
        X_train_scaled = prepared_data['X_train_scaled']
        X_test_scaled = prepared_data['X_test_scaled']
        y_train_scaled = prepared_data['y_train_scaled']
        y_test = prepared_data['y_test']
        scaler_output = prepared_data['scaler_output']
        
        # Make predictions
        y_train_pred_scaled = self.model.predict(X_train_scaled)
        y_train_pred = scaler_output.inverse_transform(y_train_pred_scaled.reshape(-1, 1))
        
        y_test_pred_scaled = self.model.predict(X_test_scaled)
        y_test_pred = scaler_output.inverse_transform(y_test_pred_scaled.reshape(-1, 1))
        
        # Calculating using raw scaled values for consistency with original code
        train_mse = mean_squared_error(y_train_scaled, y_train_pred_scaled.reshape(-1, 1))
        train_mae = mean_absolute_error(y_train_scaled, y_train_pred_scaled.reshape(-1, 1))
        train_mape = mean_absolute_percentage_error(y_train_scaled, y_train_pred_scaled.reshape(-1, 1)) * 100
        train_r2 = r2_score(y_train_scaled, y_train_pred_scaled.reshape(-1, 1))
        
        # Calculating using original values for test set
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Log metrics
        logger.info(f"For Reltime {self.reltime}:")
        logger.info("Train stats:")
        logger.info(f"MSE: {train_mse}")
        logger.info(f"MAE: {train_mae}")
        logger.info(f"MAPE: {train_mape}%")
        logger.info(f"R²: {train_r2}")
        
        logger.info("Test stats:")
        logger.info(f"MSE: {test_mse}")
        logger.info(f"MAE: {test_mae}")
        logger.info(f"MAPE: {test_mape}%")
        logger.info(f"R²: {test_r2}")
        
        # Return metrics dictionary
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'y_test': y_test,  # Add the actual y_test values
            'y_train': scaler_output.inverse_transform(y_train_scaled)  # Add the actual y_train values
        }
    
    def predict(self, X, scaler_input=None, scaler_output=None):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : array-like
            Input features
        scaler_input : sklearn.preprocessing.StandardScaler, optional
            Scaler used to transform input features
        scaler_output : sklearn.preprocessing.StandardScaler, optional
            Scaler used to inverse transform predictions
            
        Returns:
        --------
        array
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")
        
        # Scale input if scaler provided
        if scaler_input is not None:
            X_scaled = scaler_input.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Inverse transform if scaler provided
        if scaler_output is not None:
            y_pred = scaler_output.inverse_transform(y_pred_scaled.reshape(-1, 1))
        else:
            y_pred = y_pred_scaled
        
        return y_pred
    
    def save(self, circuit_name="default"):
        """
        Save the model.
        
        Parameters:
        -----------
        circuit_name : str
            Name of the circuit for the model filename
            
        Returns:
        --------
        str
            Path to the saved model
        """
        if self.model is None or self.reltime is None:
            raise ValueError("Model not trained. Call train first.")
        
        model_path = os.path.join(self.model_dir, f"reduction_{circuit_name}_reltime_{self.reltime}.lgb")
        joblib.dump(self.model, model_path)
        logger.info(f"Reduction model for reltime {self.reltime} saved to {model_path}")
        return model_path
    
    def load(self, model_path=None, circuit_name="default", reltime=None):
        """
        Load a saved model.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to the saved model file
        circuit_name : str, optional
            Name of the circuit for the model filename
        reltime : float, optional
            Reltime value for the model
            
        Returns:
        --------
        self
        """
        if model_path is None and reltime is None:
            raise ValueError("Either model_path or reltime must be provided.")
        
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"reduction_{circuit_name}_reltime_{reltime}.lgb")
            self.reltime = reltime
        
        self.model = joblib.load(model_path)
        logger.info(f"Reduction model loaded from {model_path}")
        return self


class CombinedModel:
    """Combined model that uses fresh leakage and reduction models to predict aged leakage."""
    
    def __init__(self, fresh_model=None, reduction_models=None):
        """
        Initialize the combined model.
        
        Parameters:
        -----------
        fresh_model : FreshLeakageModel, optional
            Trained fresh leakage model
        reduction_models : dict, optional
            Dictionary mapping reltime values to trained ReductionModel instances
        """
        self.fresh_model = fresh_model
        self.reduction_models = reduction_models or {}
    
    def set_fresh_model(self, fresh_model):
        """
        Set the fresh leakage model.
        
        Parameters:
        -----------
        fresh_model : FreshLeakageModel
            Trained fresh leakage model
            
        Returns:
        --------
        self
        """
        self.fresh_model = fresh_model
        return self
    
    def add_reduction_model(self, reltime, reduction_model):
        """
        Add a reduction model for a specific reltime.
        
        Parameters:
        -----------
        reltime : float
            Reltime value for the model
        reduction_model : ReductionModel
            Trained reduction model
            
        Returns:
        --------
        self
        """
        self.reduction_models[reltime] = reduction_model
        return self
    
    def predict(self, X, reltime, fresh_scalers=None, reduction_scalers=None):
        """
        Predict the aged leakage by combining fresh leakage and reduction predictions.
        
        Parameters:
        -----------
        X : array-like
            Input features
        reltime : float
            Reltime value for prediction
        fresh_scalers : dict, optional
            Dictionary containing 'input' and 'output' scalers for fresh model
        reduction_scalers : dict, optional
            Dictionary containing 'input' and 'output' scalers for reduction model
            
        Returns:
        --------
        dict
            Dictionary containing fresh leakage, reduction percentage, and aged leakage
        """
        if self.fresh_model is None:
            raise ValueError("Fresh leakage model not set. Use set_fresh_model first.")
        
        if reltime not in self.reduction_models:
            raise ValueError(f"No reduction model for reltime {reltime}. Add a model using add_reduction_model.")
        
        # Make fresh leakage prediction
        fresh_scaler_input = None if fresh_scalers is None else fresh_scalers.get('input')
        fresh_scaler_output = None if fresh_scalers is None else fresh_scalers.get('output')
        
        fresh_leakage = self.fresh_model.predict(
            X, scaler_input=fresh_scaler_input, scaler_output=fresh_scaler_output
        )
        
        # Make reduction percentage prediction
        reduction_scaler_input = None if reduction_scalers is None else reduction_scalers.get('input')
        reduction_scaler_output = None if reduction_scalers is None else reduction_scalers.get('output')
        
        reduction_percentage = self.reduction_models[reltime].predict(
            X, scaler_input=reduction_scaler_input, scaler_output=reduction_scaler_output
        )
        
        # Calculate aged leakage
        aged_leakage = fresh_leakage * (1 - reduction_percentage / 100)
        
        return {
            'fresh_leakage': fresh_leakage,
            'reduction_percentage': reduction_percentage,
            'aged_leakage': aged_leakage
        }
    
    def save(self, model_dir='./models', circuit_name='default'):
        """
        Save the model components.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save models
        circuit_name : str
            Name of the circuit for model filenames
            
        Returns:
        --------
        dict
            Dictionary containing paths to saved models
        """
        os.makedirs(model_dir, exist_ok=True)
        saved_paths = {}
        
        # Save fresh leakage model
        if self.fresh_model is not None:
            fresh_path = self.fresh_model.save(circuit_name=circuit_name)
            saved_paths['fresh_model'] = fresh_path
        
        # Save reduction models
        reduction_paths = {}
        for reltime, model in self.reduction_models.items():
            path = model.save(circuit_name=circuit_name)
            reduction_paths[reltime] = path
        
        saved_paths['reduction_models'] = reduction_paths
        
        # Save model metadata
        metadata = {
            'circuit_name': circuit_name,
            'fresh_model_path': saved_paths.get('fresh_model'),
            'reduction_model_paths': reduction_paths,
            'reltimes': list(self.reduction_models.keys()),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = os.path.join(model_dir, f"combined_model_{circuit_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        saved_paths['metadata'] = metadata_path
        
        logger.info(f"Combined model components saved. Metadata at {metadata_path}")
        return saved_paths