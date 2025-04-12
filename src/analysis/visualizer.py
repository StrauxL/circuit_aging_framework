# src/analysis/visualizer.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import logging
from src.data_processing.data_loader import CircuitDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CircuitPatternAnalyzer:
    """Class for analyzing and visualizing patterns in circuit aging data."""
    
    def __init__(self, results_dir='./results'):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        results_dir : str
            Directory to save analysis results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Set figure style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.family'] = 'serif'
        
    def analyze_leakage_reduction_patterns(self, data_loader, output_prefix=None, save_figures=True):
        """
        Analyze patterns in leakage reduction data.
        
        Parameters:
        -----------
        data_loader : CircuitDataLoader
            Data loader with loaded reduction data
        output_prefix : str, optional
            Prefix for output files
        save_figures : bool, optional
            Whether to save figures
            
        Returns:
        --------
        dict
            Dictionary containing analysis results
        """
        if data_loader.reduction_data is None:
            raise ValueError("Reduction data not loaded. Use data_loader.load_reduction_data first.")
        
        # Get unique reltimes
        reltimes = sorted(data_loader.reduction_data['reltime'].unique())
        if 0.0 in reltimes:
            reltimes.remove(0.0)  # Exclude baseline (reltime = 0)
        
        # Prepare output prefix
        if output_prefix is None:
            output_prefix = "leakage_reduction_analysis"
        
        # Dictionary to store results
        results = {
            'overall_stats': {},
            'temperature_analysis': {},
            'voltage_analysis': {},
            'feature_correlations': {},
            'pattern_instances': {}
        }
        
        # 1. Overall statistics
        logger.info("Analyzing overall leakage reduction statistics...")
        
        # Calculate statistics for each reltime
        overall_stats = pd.DataFrame()
        for rt in reltimes:
            rt_data = data_loader.reduction_data[data_loader.reduction_data['reltime'] == rt]
            stats = rt_data['leakage_reduction'].describe().to_dict()
            stats['reltime'] = rt
            overall_stats = pd.concat([overall_stats, pd.DataFrame([stats])], ignore_index=True)
        
        results['overall_stats'] = overall_stats
        
        # Create overall statistics plot
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(data=data_loader.reduction_data, x='reltime', y='leakage_reduction', ax=ax)
        ax.set_xlabel('Relative Time (seconds)', fontsize=12)
        ax.set_ylabel('Leakage Reduction (%)', fontsize=12)
        ax.set_title('Leakage Reduction Distribution by Aging Time', fontsize=14)
        
        # Convert x-axis to years for better readability (assuming seconds to years)
        seconds_per_year = 31536000  # 365 days * 24 hours * 60 minutes * 60 seconds
        ax.set_xticklabels([f"{int(float(label.get_text()) / seconds_per_year)} Years" 
                            if float(label.get_text()) > 0 else "0" 
                            for label in ax.get_xticklabels()])
        
        if save_figures:
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{output_prefix}_overall_boxplot.png"), dpi=300)
            plt.close()
        
        # 2. Temperature impact analysis
        logger.info("Analyzing temperature impact on leakage reduction...")
        
        # Get unique temperatures
        temperatures = sorted(data_loader.reduction_data['tempu'].unique())
        
        # Calculate mean reduction for each temperature and reltime
        temp_analysis = pd.DataFrame()
        for temp in temperatures:
            for rt in reltimes:
                temp_rt_data = data_loader.reduction_data[
                    (data_loader.reduction_data['tempu'] == temp) & 
                    (data_loader.reduction_data['reltime'] == rt)
                ]
                
                if not temp_rt_data.empty:
                    mean_reduction = temp_rt_data['leakage_reduction'].mean()
                    std_reduction = temp_rt_data['leakage_reduction'].std()
                    
                    temp_analysis = pd.concat([
                        temp_analysis, 
                        pd.DataFrame([{
                            'temperature': temp,
                            'reltime': rt,
                            'mean_reduction': mean_reduction,
                            'std_reduction': std_reduction
                        }])
                    ], ignore_index=True)
        
        results['temperature_analysis'] = temp_analysis
        
        # Create temperature impact visualization
        plt.figure(figsize=(14, 8))
        
        # Group by temperature
        for temp in temperatures:
            temp_data = temp_analysis[temp_analysis['temperature'] == temp]
            if not temp_data.empty:
                plt.errorbar(
                    temp_data['reltime']/seconds_per_year,  # Convert to years
                    temp_data['mean_reduction'],
                    yerr=temp_data['std_reduction'],
                    marker='o',
                    label=f"{temp}°C"
                )
        
        plt.xlabel('Aging Time (Years)', fontsize=12)
        plt.ylabel('Mean Leakage Reduction (%)', fontsize=12)
        plt.title('Temperature Impact on Leakage Reduction Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Temperature", fontsize=10)
        
        if save_figures:
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{output_prefix}_temperature_impact.png"), dpi=300)
            plt.close()
        
        # 3. Voltage impact analysis
        logger.info("Analyzing voltage impact on leakage reduction...")
        
        # Get unique voltages
        voltages = sorted(data_loader.reduction_data['pvdd'].unique())
        
        # Calculate mean reduction for each voltage and reltime
        voltage_analysis = pd.DataFrame()
        for volt in voltages:
            for rt in reltimes:
                volt_rt_data = data_loader.reduction_data[
                    (data_loader.reduction_data['pvdd'] == volt) & 
                    (data_loader.reduction_data['reltime'] == rt)
                ]
                
                if not volt_rt_data.empty:
                    mean_reduction = volt_rt_data['leakage_reduction'].mean()
                    std_reduction = volt_rt_data['leakage_reduction'].std()
                    
                    voltage_analysis = pd.concat([
                        voltage_analysis, 
                        pd.DataFrame([{
                            'voltage': volt,
                            'reltime': rt,
                            'mean_reduction': mean_reduction,
                            'std_reduction': std_reduction
                        }])
                    ], ignore_index=True)
        
        results['voltage_analysis'] = voltage_analysis
        
        # Create voltage impact visualization
        plt.figure(figsize=(14, 8))
        
        # Group by voltage
        for volt in voltages:
            volt_data = voltage_analysis[voltage_analysis['voltage'] == volt]
            if not volt_data.empty:
                plt.errorbar(
                    volt_data['reltime']/seconds_per_year,  # Convert to years
                    volt_data['mean_reduction'],
                    yerr=volt_data['std_reduction'],
                    marker='s',
                    label=f"{volt}V"
                )
        
        plt.xlabel('Aging Time (Years)', fontsize=12)
        plt.ylabel('Mean Leakage Reduction (%)', fontsize=12)
        plt.title('Supply Voltage Impact on Leakage Reduction Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Voltage", fontsize=10)
        
        if save_figures:
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{output_prefix}_voltage_impact.png"), dpi=300)
            plt.close()
        
        # 4. Feature correlations
        logger.info("Analyzing feature correlations with leakage reduction...")
        
        # Select numerical features for correlation analysis
        feature_cols = [col for col in data_loader.reduction_data.columns 
                       if col not in ['reltime', 'leakage', 'baseline_leakage', 'leakage_reduction']]
        
        # Calculate correlations for each reltime
        feature_correlations = {}
        for rt in reltimes:
            rt_data = data_loader.reduction_data[data_loader.reduction_data['reltime'] == rt]
            
            # Calculate correlation with leakage_reduction
            correlations = rt_data[feature_cols + ['leakage_reduction']].corr()['leakage_reduction'].drop('leakage_reduction')
            feature_correlations[rt] = correlations.sort_values(ascending=False)
        
        results['feature_correlations'] = feature_correlations
        
        # Visualize top correlations for selected reltimes (e.g., 1, 5, 10 years)
        selected_years = [1, 5, 10]
        selected_reltimes = [year * seconds_per_year for year in selected_years 
                             if year * seconds_per_year in reltimes]
        
        if selected_reltimes:
            plt.figure(figsize=(16, 10))
            
            # Plot correlations for top N features
            top_n = 10
            for i, rt in enumerate(selected_reltimes):
                corrs = feature_correlations[rt].abs().sort_values(ascending=False).head(top_n)
                
                plt.subplot(len(selected_reltimes), 1, i+1)
                corrs.plot(kind='barh', color=plt.cm.viridis(i/len(selected_reltimes)))
                
                years = rt / seconds_per_year
                plt.title(f'Feature Correlation with Leakage Reduction (Year {int(years)})', fontsize=12)
                plt.xlabel('Correlation Coefficient (Absolute Value)', fontsize=10)
                plt.tight_layout()
            
            if save_figures:
                plt.savefig(os.path.join(self.results_dir, f"{output_prefix}_feature_correlations.png"), dpi=300)
                plt.close()
        
        # 5. Find and visualize specific pattern instances
        logger.info("Finding pattern instances...")
        
        # Use existing method to find patterns
        pattern_instances = data_loader.find_pattern_instances(num_features_to_match=21, num_samples=100)
        
        # Group patterns by their identifying features (temperature and voltage)
        pattern_groups = []
        
        # For simplicity, group by temperature and voltage
        for temp in temperatures:
            for volt in voltages:
                temp_volt_mask = (pattern_instances['tempu'] == temp) & (pattern_instances['pvdd'] == volt)
                temp_volt_data = pattern_instances[temp_volt_mask]
                
                if not temp_volt_data.empty:
                    # Find all unique patterns with this temp and volt
                    unique_pattern_starts = []
                    
                    for i in range(len(temp_volt_data)):
                        # Check if this is the start of a new pattern
                        if i == 0 or temp_volt_data.iloc[i]['reltime'] == 0.0:
                            unique_pattern_starts.append(i)
                    
                    # Extract and plot each unique pattern
                    for start_idx in unique_pattern_starts:
                        # Find all rows with matching first 21 columns
                        key_values = temp_volt_data.iloc[start_idx, :21]
                        mask = temp_volt_data.iloc[:, :21].eq(key_values).all(axis=1)
                        pattern_data = temp_volt_data[mask].sort_values('reltime')
                        
                        if len(pattern_data) > 1:  # Only consider patterns with multiple time points
                            pattern_groups.append({
                                'temperature': temp,
                                'voltage': volt,
                                'data': pattern_data.reset_index(drop=True)
                            })
        
        results['pattern_instances'] = pattern_groups
        
        # Visualize selected pattern groups
        for i, pattern in enumerate(pattern_groups[:min(10, len(pattern_groups))]):
            plt.figure(figsize=(10, 6))
            
            pattern_data = pattern['data']
            plt.plot(
                pattern_data['reltime']/seconds_per_year,
                pattern_data['leakage_reduction'],
                'o-',
                lw=2,
                label=f"Temp={pattern['temperature']}°C, Volt={pattern['voltage']}V"
            )
            
            plt.xlabel('Aging Time (Years)', fontsize=12)
            plt.ylabel('Leakage Reduction (%)', fontsize=12)
            plt.title(f'Leakage Reduction Pattern (T={pattern["temperature"]}°C, V={pattern["voltage"]}V)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            if save_figures:
                plt.tight_layout()
                plt.savefig(os.path.join(
                    self.results_dir, 
                    f"{output_prefix}_pattern_{i}_T{pattern['temperature']}_V{pattern['voltage']}.png"
                ), dpi=300)
                plt.close()
        
        # Create a comprehensive pattern comparison
        if pattern_groups:
            plt.figure(figsize=(12, 8))
            
            for i, pattern in enumerate(pattern_groups[:min(6, len(pattern_groups))]):
                pattern_data = pattern['data']
                plt.plot(
                    pattern_data['reltime']/seconds_per_year,
                    pattern_data['leakage_reduction'],
                    'o-',
                    lw=2,
                    label=f"T={pattern['temperature']}°C, V={pattern['voltage']}V"
                )
            
            plt.xlabel('Aging Time (Years)', fontsize=12)
            plt.ylabel('Leakage Reduction (%)', fontsize=12)
            plt.title('Comparison of Leakage Reduction Patterns', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(title="Circuit Parameters")
            
            if save_figures:
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f"{output_prefix}_pattern_comparison.png"), dpi=300)
                plt.close()
        
        logger.info("Leakage reduction pattern analysis completed.")
        return results
    
    def analyze_model_performance(self, fresh_metrics, reduction_metrics_by_reltime, output_prefix=None, save_figures=True):
        """
        Analyze model performance metrics.
        
        Parameters:
        -----------
        fresh_metrics : dict
            Dictionary containing fresh model evaluation metrics
        reduction_metrics_by_reltime : dict
            Dictionary mapping reltimes to reduction model evaluation metrics
        output_prefix : str, optional
            Prefix for output files
        save_figures : bool, optional
            Whether to save figures
            
        Returns:
        --------
        dict
            Dictionary containing analysis results
        """
        # Prepare output prefix
        if output_prefix is None:
            output_prefix = "model_performance_analysis"
        
        # Dictionary to store results
        results = {
            'fresh_model': fresh_metrics,
            'reduction_models': reduction_metrics_by_reltime,
            'metrics_comparison': {}
        }
        
        logger.info("Analyzing model performance...")
        
        # 1. Visualize fresh model performance
        plt.figure(figsize=(12, 7))
        
        # Create scatter plot of actual vs predicted values
        plt.scatter(
            fresh_metrics['y_test'], 
            fresh_metrics['y_test_pred'], 
            alpha=0.5,
            label="Test Data"
        )
        
        # Add perfect prediction line
        min_val = min(min(fresh_metrics['y_test']), min(fresh_metrics['y_test_pred']))
        max_val = max(max(fresh_metrics['y_test']), max(fresh_metrics['y_test_pred']))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Prediction")
        
        plt.xlabel('Actual Leakage', fontsize=12)
        plt.ylabel('Predicted Leakage', fontsize=12)
        plt.title('Fresh Leakage Model: Actual vs Predicted', fontsize=14)
        
        # Add metrics as text annotation
        textstr = '\n'.join([
            f"Test MSE: {fresh_metrics['test_mse']:.2e}",
            f"Test MAE: {fresh_metrics['test_mae']:.2e}",
            f"Test MAPE: {fresh_metrics['test_mape']:.2f}%",
            f"Test R²: {fresh_metrics['test_r2']:.4f}"
        ])
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                     verticalalignment='top', bbox=props)
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_figures:
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{output_prefix}_fresh_model_performance.png"), dpi=300)
            plt.close()
        
        # 2. Visualize reduction model performance for each reltime
        seconds_per_year = 31536000
        
        # Extract metrics for comparison
        reltimes = sorted(reduction_metrics_by_reltime.keys())
        metrics_comparison = {
            'reltime': [],
            'reltime_years': [],
            'test_mse': [],
            'test_mae': [],
            'test_mape': [],
            'test_r2': []
        }
        
        for rt in reltimes:
            metrics = reduction_metrics_by_reltime[rt]
            
            # Add to comparison data
            metrics_comparison['reltime'].append(rt)
            metrics_comparison['reltime_years'].append(rt / seconds_per_year)
            metrics_comparison['test_mse'].append(metrics['test_mse'])
            metrics_comparison['test_mae'].append(metrics['test_mae'])
            metrics_comparison['test_mape'].append(metrics['test_mape'])
            metrics_comparison['test_r2'].append(metrics['test_r2'])
            
            # Create scatter plot for this reltime
            plt.figure(figsize=(10, 6))
            
            plt.scatter(
                metrics['y_test'], 
                metrics['y_test_pred'], 
                alpha=0.5,
                label="Test Data"
            )
            
            # Add perfect prediction line
            min_val = min(min(metrics['y_test']), min(metrics['y_test_pred']))
            max_val = max(max(metrics['y_test']), max(metrics['y_test_pred']))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Prediction")
            
            plt.xlabel('Actual Reduction (%)', fontsize=12)
            plt.ylabel('Predicted Reduction (%)', fontsize=12)
            plt.title(f'Reduction Model (Year {int(rt/seconds_per_year)}): Actual vs Predicted', fontsize=14)
            
            # Add metrics as text annotation
            textstr = '\n'.join([
                f"Test MSE: {metrics['test_mse']:.2e}",
                f"Test MAE: {metrics['test_mae']:.2e}",
                f"Test MAPE: {metrics['test_mape']:.2f}%",
                f"Test R²: {metrics['test_r2']:.4f}"
            ])
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction', 
                        verticalalignment='top', bbox=props)
            
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            if save_figures:
                plt.tight_layout()
                plt.savefig(os.path.join(
                    self.results_dir, 
                    f"{output_prefix}_reduction_model_year{int(rt/seconds_per_year)}_performance.png"
                ), dpi=300)
                plt.close()
        
        results['metrics_comparison'] = metrics_comparison
        
        # 3. Create comparison plot of metrics across reltimes
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # MSE
        axs[0, 0].plot(metrics_comparison['reltime_years'], metrics_comparison['test_mse'], 'o-', lw=2)
        axs[0, 0].set_xlabel('Aging Time (Years)', fontsize=12)
        axs[0, 0].set_ylabel('Test MSE', fontsize=12)
        axs[0, 0].set_title('Mean Squared Error by Aging Time', fontsize=12)
        axs[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axs[0, 1].plot(metrics_comparison['reltime_years'], metrics_comparison['test_mae'], 'o-', lw=2, color='orange')
        axs[0, 1].set_xlabel('Aging Time (Years)', fontsize=12)
        axs[0, 1].set_ylabel('Test MAE', fontsize=12)
        axs[0, 1].set_title('Mean Absolute Error by Aging Time', fontsize=12)
        axs[0, 1].grid(True, alpha=0.3)
        
        # MAPE
        axs[1, 0].plot(metrics_comparison['reltime_years'], metrics_comparison['test_mape'], 'o-', lw=2, color='green')
        axs[1, 0].set_xlabel('Aging Time (Years)', fontsize=12)
        axs[1, 0].set_ylabel('Test MAPE (%)', fontsize=12)
        axs[1, 0].set_title('Mean Absolute Percentage Error by Aging Time', fontsize=12)
        axs[1, 0].grid(True, alpha=0.3)
        
        # R²
        axs[1, 1].plot(metrics_comparison['reltime_years'], metrics_comparison['test_r2'], 'o-', lw=2, color='red')
        axs[1, 1].set_xlabel('Aging Time (Years)', fontsize=12)
        axs[1, 1].set_ylabel('Test R²', fontsize=12)
        axs[1, 1].set_title('R² Score by Aging Time', fontsize=12)
        axs[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Model Performance Metrics Across Aging Times', fontsize=16)
        
        if save_figures:
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
            plt.savefig(os.path.join(self.results_dir, f"{output_prefix}_metrics_comparison.png"), dpi=300)
            plt.close()
        
        logger.info("Model performance analysis completed.")
        return results