# src/data_processing/data_loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CircuitDataLoader:
    """Class for loading and preprocessing circuit data."""
    
    def __init__(self, data_dir='./data'):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the dataset files
        """
        self.data_dir = data_dir
        self.fresh_data = None
        self.reduction_data = None
        self.scalers = {}
        
    def load_fresh_simulation_data(self, file_path=None, filename=None):
        """
        Load fresh simulation data.
        
        Parameters:
        -----------
        file_path : str, optional
            Full path to the fresh simulation data file
        filename : str, optional
            Name of the file in the data directory
            
        Returns:
        --------
        pandas.DataFrame
            Loaded fresh simulation data
        """
        if file_path is None and filename is None:
            raise ValueError("Either file_path or filename must be provided")
        
        path = file_path if file_path else os.path.join(self.data_dir, filename)
        logger.info(f"Loading fresh simulation data from {path}")
        
        try:
            data = pd.read_csv(path)
            
            # Remove unnecessary columns if they exist
            cols_to_drop = ['Unnamed: 0', 'index', 'alter#']
            for col in cols_to_drop:
                if col in data.columns:
                    data = data.drop(col, axis=1)
                    
            self.fresh_data = data
            logger.info(f"Successfully loaded fresh simulation data with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading fresh simulation data: {str(e)}")
            raise
            
    def load_reduction_data(self, file_path=None, filename=None):
        """
        Load leakage reduction data.
        
        Parameters:
        -----------
        file_path : str, optional
            Full path to the reduction data file
        filename : str, optional
            Name of the file in the data directory
            
        Returns:
        --------
        pandas.DataFrame
            Loaded reduction data
        """
        if file_path is None and filename is None:
            raise ValueError("Either file_path or filename must be provided")
        
        path = file_path if file_path else os.path.join(self.data_dir, filename)
        logger.info(f"Loading reduction data from {path}")
        
        try:
            data = pd.read_csv(path)
            
            # Remove unnecessary columns if they exist
            if 'Unnamed: 0' in data.columns:
                data = data.drop(['Unnamed: 0'], axis=1)
                
            self.reduction_data = data
            logger.info(f"Successfully loaded reduction data with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading reduction data: {str(e)}")
            raise
    
    def prepare_fresh_simulation_data(self, input_features=None, output_features=None, test_size=0.1, random_state=1):
        """
        Prepare fresh simulation data for model training.
        
        Parameters:
        -----------
        input_features : list, optional
            List of input feature names. If None, uses default features.
        output_features : list, optional
            List of output feature names. If None, uses 'leakage'.
        test_size : float, optional
            Proportion of the dataset to include in the test split.
        random_state : int, optional
            Random seed for reproducibility.
            
        Returns:
        --------
        tuple
            (X_train_scaled, X_test_scaled, y_train_scaled, y_test, scaler_input, scaler_output)
        """
        if self.fresh_data is None:
            raise ValueError("Fresh simulation data not loaded. Call load_fresh_simulation_data first.")
        
        # Default features if not provided
        if input_features is None:
            input_features = ['vin_a', 'vin_b', 'vin_c', 'tempu', 'pvdd', 'cqload', 'nc0subn', 'nc0subp', 
                             'nbodyn', 'nbodyp', 'ni0subn', 'ni0subp', 'toxpn', 'toxpp', 'nsdn', 'nsdp', 
                             'tfinn', 'tfinp', 'hfinn', 'hfinp', 'eotn', 'eotp', 'lg']
        
        if output_features is None:
            output_features = ['leakage']
        
        # Check if all features exist in the data
        missing_inputs = [f for f in input_features if f not in self.fresh_data.columns]
        missing_outputs = [f for f in output_features if f not in self.fresh_data.columns]
        
        if missing_inputs:
            logger.warning(f"Missing input features: {missing_inputs}")
            input_features = [f for f in input_features if f in self.fresh_data.columns]
            
        if missing_outputs:
            logger.warning(f"Missing output features: {missing_outputs}")
            output_features = [f for f in output_features if f in self.fresh_data.columns]
        
        # Split the data
        X = self.fresh_data[input_features].values
        y = self.fresh_data[output_features].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        # Scale the data
        scaler_input = StandardScaler()
        scaler_output = StandardScaler()
        
        X_train_scaled = scaler_input.fit_transform(X_train)
        y_train_scaled = scaler_output.fit_transform(y_train.reshape(-1, 1))
        X_test_scaled = scaler_input.transform(X_test)
        
        # Store the scalers
        self.scalers['fresh_input'] = scaler_input
        self.scalers['fresh_output'] = scaler_output
        
        return {
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train_scaled': y_train_scaled,
            'y_test': y_test,
            'input_features': input_features,
            'output_features': output_features,
            'scaler_input': scaler_input,
            'scaler_output': scaler_output
        }
    
    def prepare_reduction_data(self, reltime, input_features=None, output_features=None, test_size=0.1, random_state=5):
        """
        Prepare reduction data for a specific reltime for model training.
        
        Parameters:
        -----------
        reltime : float
            The reltime value to filter data for
        input_features : list, optional
            List of input feature names. If None, uses default features.
        output_features : list, optional
            List of output feature names. If None, uses 'leakage_reduction'.
        test_size : float, optional
            Proportion of the dataset to include in the test split.
        random_state : int, optional
            Random seed for reproducibility.
            
        Returns:
        --------
        dict
            Dictionary containing prepared data
        """
        if self.reduction_data is None:
            raise ValueError("Reduction data not loaded. Call load_reduction_data first.")
        
        # Default features if not provided
        if input_features is None:
            input_features = ['vin_a', 'vin_b', 'vin_c', 'tempu', 'pvdd', 'nc0subn', 'nc0subp', 
                             'nbodyn', 'nbodyp', 'ni0subn', 'ni0subp', 'toxpn', 'toxpp', 'nsdn', 
                             'nsdp', 'tfinn', 'tfinp', 'hfinn', 'hfinp', 'eotn', 'eotp', 'lg']
        
        if output_features is None:
            output_features = ['leakage_reduction']
        
        # Check if reltime exists
        if reltime not in self.reduction_data['reltime'].unique():
            available_reltimes = self.reduction_data['reltime'].unique()
            raise ValueError(f"Reltime {reltime} not found in the data. Available reltimes: {available_reltimes}")
        
        # Filter data for the specific reltime
        data_model = self.reduction_data.loc[self.reduction_data['reltime'] == reltime]
        
        # Check if all features exist in the data
        missing_inputs = [f for f in input_features if f not in data_model.columns]
        missing_outputs = [f for f in output_features if f not in data_model.columns]
        
        if missing_inputs:
            logger.warning(f"Missing input features: {missing_inputs}")
            input_features = [f for f in input_features if f in data_model.columns]
            
        if missing_outputs:
            logger.warning(f"Missing output features: {missing_outputs}")
            output_features = [f for f in output_features if f in data_model.columns]
        
        # Split the data
        X = data_model[input_features].values
        y = data_model[output_features].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        # Scale the data
        scaler_input = StandardScaler()
        scaler_output = StandardScaler()
        
        X_train_scaled = scaler_input.fit_transform(X_train)
        y_train_scaled = scaler_output.fit_transform(y_train.reshape(-1, 1))
        X_test_scaled = scaler_input.transform(X_test)
        
        # Store the scalers
        key = f'reduction_input_{reltime}'
        self.scalers[key] = scaler_input
        key = f'reduction_output_{reltime}'
        self.scalers[key] = scaler_output
        
        return {
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train_scaled': y_train_scaled,
            'y_test': y_test,
            'input_features': input_features,
            'output_features': output_features,
            'scaler_input': scaler_input,
            'scaler_output': scaler_output,
            'reltime': reltime
        }
    
    def find_pattern_instances(self, num_features_to_match=21, sample_indices=None, num_samples=100):
        """
        Find patterns in the reduction data by matching feature values.
        
        Parameters:
        -----------
        num_features_to_match : int, optional
            Number of initial features to match (default: 21, matching all circuit parameters)
        sample_indices : list, optional
            List of sample indices to use as reference. If None, uses range(num_samples).
        num_samples : int, optional
            Number of samples to use as reference if sample_indices is None.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing all matching patterns
        """
        if self.reduction_data is None:
            raise ValueError("Reduction data not loaded. Call load_reduction_data first.")
        
        # Get indices to sample
        if sample_indices is None:
            sample_indices = list(range(min(num_samples, len(self.reduction_data))))
        
        # Create result dataframe
        result_df = pd.DataFrame()
        
        # Process each sample
        for idx in sample_indices:
            # Create mask for matching features
            mask = self.reduction_data.iloc[:, :num_features_to_match].eq(
                self.reduction_data.iloc[idx, :num_features_to_match]
            ).all(axis=1)
            
            # Get filtered data
            filtered_data = self.reduction_data[mask]
            
            # Add to result if not empty
            if not filtered_data.empty:
                if result_df.empty:
                    result_df = filtered_data
                else:
                    # Check if this pattern is already in result_df
                    pattern_key = tuple(filtered_data.iloc[0, :num_features_to_match])
                    if not any(tuple(result_df.iloc[i, :num_features_to_match]) == pattern_key 
                               for i in range(len(result_df)) if i % len(filtered_data) == 0):
                        result_df = pd.concat([result_df, filtered_data], ignore_index=True)
        
        return result_df