# src/symbolic_regression/pattern_finder.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sympy import symbols, lambdify
from pysr import PySRRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LeakagePatternFinder:
    """Class for finding mathematical patterns in leakage reduction data using symbolic regression."""
    
    def __init__(self, results_dir='./results'):
        """
        Initialize the pattern finder.
        
        Parameters:
        -----------
        results_dir : str
            Directory to save results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def find_time_dependent_pattern(self, pattern_data, output_prefix=None, save_figures=True):
        """
        Find a mathematical formula for the time-dependent leakage reduction pattern.
        
        Parameters:
        -----------
        pattern_data : pandas.DataFrame
            DataFrame containing reltime and leakage_reduction columns
        output_prefix : str, optional
            Prefix for output files
        save_figures : bool, optional
            Whether to save figures
            
        Returns:
        --------
        dict
            Dictionary containing the best formula and related information
        """
        if pattern_data.empty or 'reltime' not in pattern_data.columns or 'leakage_reduction' not in pattern_data.columns:
            raise ValueError("Pattern data must contain 'reltime' and 'leakage_reduction' columns")
        
        # Prepare output prefix
        if output_prefix is None:
            output_prefix = "time_pattern"
        
        logger.info("Finding time-dependent pattern using symbolic regression...")
        
        # Convert reltime to years for better numerical stability
        seconds_per_year = 31536000
        X = pattern_data['reltime'].values.reshape(-1, 1) / seconds_per_year
        y = pattern_data['leakage_reduction'].values
        
        # Set up the symbolic regression model
        model = PySRRegressor(
            niterations=100,  # Increase for better results, but will take longer
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log", "sqrt"],
            loss="loss(x, y) = (x - y)^2",  # Mean squared error loss
            maxsize=15,  # Maximum complexity of equations
            populations=15,  # Number of parallel populations
            population_size=30,  # Population size
            verbosity=0,  # Quiet mode
            batching=False,  # No batching
            parsimony=0.001  # Slight preference for simpler equations
        )
        
        try:
            # Fit the model
            model.fit(X, y, variable_names=["years"])
            
            # Get the best equation
            equations = model.equations_
            
            if equations.empty:
                logger.warning("No equations found.")
                return {"success": False, "message": "No equations found"}
            
            best_eq = equations.iloc[0]
            best_formula = best_eq["sympy_format"]
            complexity = best_eq["complexity"]
            score = best_eq["score"]
            
            logger.info(f"Best formula: {best_formula} (Complexity: {complexity}, Score: {score})")
            
            # Convert the symbolic formula to a callable function
            x = symbols('years')
            formula_callable = lambdify(x, best_formula)
            
            # Generate predictions for plotting
            X_plot = np.linspace(0, max(X.flatten()) * 1.1, 100)
            y_plot = np.array([formula_callable(xi) for xi in X_plot])
            
            # Plot the original data and the fitted curve
            plt.figure(figsize=(12, 7))
            plt.scatter(X, y, label='Original Data', color='blue', alpha=0.7)
            plt.plot(X_plot, y_plot, label=f'Fitted Model: {best_formula}', color='red', linewidth=2)
            
            plt.xlabel('Aging Time (Years)', fontsize=12)
            plt.ylabel('Leakage Reduction (%)', fontsize=12)
            plt.title('Symbolic Regression: Leakage Reduction over Time', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add the formula as text annotation
            textstr = f"Formula: {best_formula}\nComplexity: {complexity}\nScore: {score:.4f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                         verticalalignment='top', bbox=props)
            
            if save_figures:
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f"{output_prefix}_symbolic_regression.png"), dpi=300)
                plt.close()
            
            # Calculate prediction errors
            y_pred = np.array([formula_callable(xi) for xi in X.flatten()])
            mse = np.mean((y - y_pred) ** 2)
            mae = np.mean(np.abs(y - y_pred))
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            logger.info(f"Model evaluation - MSE: {mse}, MAE: {mae}, R²: {r2}")
            
            # Return the results
            return {
                "success": True,
                "formula": str(best_formula),
                "sympy_formula": best_formula,
                "complexity": complexity,
                "score": score,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "original_data": {"X": X, "y": y},
                "predictions": {"X": X_plot, "y": y_plot}
            }
            
        except Exception as e:
            logger.error(f"Error in symbolic regression: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def find_parameter_patterns(self, data_loader, parameter, reltime=None, output_prefix=None, save_figures=True):
        """
        Find patterns in how a specific parameter affects leakage reduction.
        
        Parameters:
        -----------
        data_loader : CircuitDataLoader
            Data loader with loaded reduction data
        parameter : str
            Name of the parameter to analyze
        reltime : float, optional
            Specific reltime to analyze. If None, uses the maximum reltime.
        output_prefix : str, optional
            Prefix for output files
        save_figures : bool, optional
            Whether to save figures
            
        Returns:
        --------
        dict
            Dictionary containing the best formula and related information
        """
        if data_loader.reduction_data is None:
            raise ValueError("Reduction data not loaded. Use data_loader.load_reduction_data first.")
        
        if parameter not in data_loader.reduction_data.columns:
            raise ValueError(f"Parameter '{parameter}' not found in data columns")
        
        # Prepare output prefix
        if output_prefix is None:
            output_prefix = f"parameter_{parameter}"
        
        # Get unique reltimes
        reltimes = sorted(data_loader.reduction_data['reltime'].unique())
        if 0.0 in reltimes:
            reltimes.remove(0.0)  # Exclude baseline (reltime = 0)
        
        # Select reltime
        if reltime is None:
            reltime = max(reltimes)
        elif reltime not in reltimes:
            closest_reltime = min(reltimes, key=lambda x: abs(x - reltime))
            logger.warning(f"Reltime {reltime} not found. Using closest reltime: {closest_reltime}")
            reltime = closest_reltime
        
        # Filter data for the selected reltime
        filtered_data = data_loader.reduction_data[data_loader.reduction_data['reltime'] == reltime]
        
        # Group by the parameter and calculate mean reduction
        grouped_data = filtered_data.groupby(parameter)['leakage_reduction'].mean().reset_index()
        
        logger.info(f"Finding pattern for parameter '{parameter}' at reltime {reltime}...")
        
        # Check if we have enough data points
        if len(grouped_data) < 5:
            logger.warning(f"Not enough data points for parameter '{parameter}' (only {len(grouped_data)} unique values)")
            return {"success": False, "message": f"Not enough data points for parameter '{parameter}'"}
        
        # Set up the symbolic regression model
        X = grouped_data[parameter].values.reshape(-1, 1)
        y = grouped_data['leakage_reduction'].values
        
        model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log", "sqrt"],
            loss="loss(x, y) = (x - y)^2",
            maxsize=15,
            populations=15,
            population_size=30,
            verbosity=0,
            batching=False,
            parsimony=0.001
        )
        
        try:
            # Fit the model
            param_name = parameter
            model.fit(X, y, variable_names=[param_name])
            
            # Get the best equation
            equations = model.equations_
            
            if equations.empty:
                logger.warning("No equations found.")
                return {"success": False, "message": "No equations found"}
            
            best_eq = equations.iloc[0]
            best_formula = best_eq["sympy_format"]
            complexity = best_eq["complexity"]
            score = best_eq["score"]
            
            logger.info(f"Best formula: {best_formula} (Complexity: {complexity}, Score: {score})")
            
            # Convert the symbolic formula to a callable function
            x = symbols(param_name)
            formula_callable = lambdify(x, best_formula)
            
            # Generate predictions for plotting
            X_min, X_max = min(X.flatten()), max(X.flatten())
            X_plot = np.linspace(X_min * 0.9, X_max * 1.1, 100)
            y_plot = np.array([formula_callable(xi) for xi in X_plot])
            
            # Plot the original data and the fitted curve
            plt.figure(figsize=(12, 7))
            plt.scatter(X, y, label='Data Points', color='blue', alpha=0.7)
            plt.plot(X_plot, y_plot, label=f'Fitted Model: {best_formula}', color='red', linewidth=2)
            
            seconds_per_year = 31536000
            years = reltime / seconds_per_year
            
            plt.xlabel(f'{parameter}', fontsize=12)
            plt.ylabel('Leakage Reduction (%)', fontsize=12)
            plt.title(f'Parameter {parameter} Impact on Leakage Reduction (Year {int(years)})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add the formula as text annotation
            textstr = f"Formula: {best_formula}\nComplexity: {complexity}\nScore: {score:.4f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                         verticalalignment='top', bbox=props)
            
            if save_figures:
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f"{output_prefix}_year{int(years)}_symbolic_regression.png"), dpi=300)
                plt.close()
            
            # Calculate prediction errors
            y_pred = np.array([formula_callable(xi) for xi in X.flatten()])
            mse = np.mean((y - y_pred) ** 2)
            mae = np.mean(np.abs(y - y_pred))
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            logger.info(f"Model evaluation - MSE: {mse}, MAE: {mae}, R²: {r2}")
            
            # Return the results
            return {
                "success": True,
                "formula": str(best_formula),
                "sympy_formula": best_formula,
                "complexity": complexity,
                "score": score,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "original_data": {"X": X, "y": y},
                "predictions": {"X": X_plot, "y": y_plot},
                "parameter": parameter,
                "reltime": reltime,
                "years": years
            }
            
        except Exception as e:
            logger.error(f"Error in symbolic regression for parameter '{parameter}': {str(e)}")
            return {"success": False, "message": str(e)}