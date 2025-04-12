# src/main.py

import os
import argparse
import logging
import pandas as pd
import numpy as np
from src.data_processing.data_loader import CircuitDataLoader
from src.modeling.models import FreshLeakageModel, ReductionModel, CombinedModel
from src.analysis.visualizer import CircuitPatternAnalyzer
from src.symbolic_regression.pattern_finder import LeakagePatternFinder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_circuit(fresh_data_path, reduction_data_path, circuit_name, output_dir=None, train_models=True):
    """
    Process a circuit's data, train models, and analyze patterns.
    
    Parameters:
    -----------
    fresh_data_path : str
        Path to fresh simulation data
    reduction_data_path : str
        Path to reduction data
    circuit_name : str
        Name of the circuit (e.g., 'NAND3')
    output_dir : str, optional
        Output directory for results
    train_models : bool, optional
        Whether to train new models or load existing ones
        
    Returns:
    --------
    dict
        Dictionary containing processing results
    """
    # Set up directories
    if output_dir is None:
        output_dir = f"./results/{circuit_name}"
    
    models_dir = f"./models/{circuit_name}"
    results_dir = f"{output_dir}/analysis"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"Processing circuit: {circuit_name}")
    logger.info(f"Fresh data path: {fresh_data_path}")
    logger.info(f"Reduction data path: {reduction_data_path}")
    
    # Initialize data loader
    data_loader = CircuitDataLoader()
    
    # Load data
    fresh_data = data_loader.load_fresh_simulation_data(file_path=fresh_data_path)
    reduction_data = data_loader.load_reduction_data(file_path=reduction_data_path)
    
    # 1. Train/load fresh leakage model
    fresh_model = FreshLeakageModel(model_dir=models_dir)
    fresh_model_path = os.path.join(models_dir, f"fresh_leakage_{circuit_name}.cbm")
    
    if train_models or not os.path.exists(fresh_model_path):
        logger.info("Training fresh leakage model...")
        
        # Prepare data
        fresh_data_prepared = data_loader.prepare_fresh_simulation_data()
        
        # Train model
        fresh_model.train(fresh_data_prepared)
        
        # Save model
        fresh_model.save(circuit_name=circuit_name)
        
        # Evaluate model
        fresh_metrics = fresh_model.evaluate(fresh_data_prepared)
    else:
        logger.info(f"Loading existing fresh leakage model from {fresh_model_path}")
        fresh_model.load(circuit_name=circuit_name)
        
        # Prepare data for evaluation
        fresh_data_prepared = data_loader.prepare_fresh_simulation_data()
        
        # Evaluate model
        fresh_metrics = fresh_model.evaluate(fresh_data_prepared)
    
    # 2. Train/load reduction models for each reltime
    reltimes = sorted(reduction_data['reltime'].unique())
    if 0.0 in reltimes:
        reltimes.remove(0.0)  # Exclude baseline (reltime = 0)
    
    reduction_models = {}
    reduction_metrics = {}
    
    for rt in reltimes:
        rt_model = ReductionModel(model_dir=models_dir)
        rt_model_path = os.path.join(models_dir, f"reduction_{circuit_name}_reltime_{rt}.lgb")
        
        if train_models or not os.path.exists(rt_model_path):
            logger.info(f"Training reduction model for reltime {rt}...")
            
            # Prepare data
            rt_data_prepared = data_loader.prepare_reduction_data(rt)
            
            # Train model
            rt_model.train(rt_data_prepared)
            
            # Save model
            rt_model.save(circuit_name=circuit_name)
            
            # Evaluate model
            rt_metrics = rt_model.evaluate(rt_data_prepared)
        else:
            logger.info(f"Loading existing reduction model from {rt_model_path}")
            rt_model.load(model_path=rt_model_path)
            
            # Prepare data for evaluation
            rt_data_prepared = data_loader.prepare_reduction_data(rt)
            
            # Evaluate model
            rt_metrics = rt_model.evaluate(rt_data_prepared)
        
        reduction_models[rt] = rt_model
        reduction_metrics[rt] = rt_metrics
    
    # 3. Create combined model
    combined_model = CombinedModel(fresh_model, reduction_models)
    
    # 4. Analyze leakage reduction patterns
    analyzer = CircuitPatternAnalyzer(results_dir=results_dir)
    pattern_analysis = analyzer.analyze_leakage_reduction_patterns(
        data_loader, output_prefix=f"{circuit_name}_leakage_reduction"
    )
    
    print(f"############# {fresh_metrics} ########################")
    # 5. Analyze model performance
    model_analysis = analyzer.analyze_model_performance(
        fresh_metrics, reduction_metrics, output_prefix=f"{circuit_name}_model_performance"
    )
    
    # 6. Find mathematical patterns using symbolic regression
    pattern_finder = LeakagePatternFinder(results_dir=results_dir)
    
    # Find pattern instances
    pattern_instances = data_loader.find_pattern_instances(num_features_to_match=21, num_samples=100)
    
    # Group patterns by tempu and pvdd
    pattern_groups = []
    for temp in pattern_instances['tempu'].unique():
        for volt in pattern_instances['pvdd'].unique():
            temp_volt_mask = (pattern_instances['tempu'] == temp) & (pattern_instances['pvdd'] == volt)
            temp_volt_data = pattern_instances[temp_volt_mask]
            
            if not temp_volt_data.empty:
                # Find all unique patterns with this temp and volt
                unique_patterns = []
                for i, row in temp_volt_data.iterrows():
                    if row['reltime'] == 0.0:  # Start of a pattern
                        key_values = row.iloc[:21]
                        mask = temp_volt_data.iloc[:, :21].eq(key_values).all(axis=1)
                        pattern_data = temp_volt_data[mask].sort_values('reltime')
                        
                        if len(pattern_data) > 1:  # Only consider patterns with multiple time points
                            unique_patterns.append(pattern_data.reset_index(drop=True))
                
                for i, pattern in enumerate(unique_patterns):
                    pattern_groups.append({
                        'temperature': temp,
                        'voltage': volt,
                        'data': pattern,
                        'pattern_id': f"T{temp}_V{volt}_{i}"
                    })
    
    # Apply symbolic regression to each pattern group
    symbolic_results = []
    for pattern in pattern_groups:
        pattern_id = pattern['pattern_id']
        pattern_data = pattern['data']
        
        logger.info(f"Finding mathematical pattern for {pattern_id}...")
        result = pattern_finder.find_time_dependent_pattern(
            pattern_data, output_prefix=f"{circuit_name}_pattern_{pattern_id}"
        )
        
        if result['success']:
            result['pattern_id'] = pattern_id
            result['temperature'] = pattern['temperature']
            result['voltage'] = pattern['voltage']
            symbolic_results.append(result)
    
    # 7. Find parameter-specific patterns
    parameter_results = {}
    important_parameters = ['tempu', 'pvdd', 'lg']  # Add more parameters as needed
    
    for param in important_parameters:
        if param in reduction_data.columns:
            # Find pattern for maximum reltime (10 years typically)
            max_reltime = max(reltimes)
            result = pattern_finder.find_parameter_patterns(
                data_loader, param, reltime=max_reltime, 
                output_prefix=f"{circuit_name}_param_{param}"
            )
            parameter_results[param] = result
    
    # Return all results
    return {
        'circuit_name': circuit_name,
        'data_loader': data_loader,
        'fresh_model': fresh_model,
        'reduction_models': reduction_models,
        'combined_model': combined_model,
        'fresh_metrics': fresh_metrics,
        'reduction_metrics': reduction_metrics,
        'pattern_analysis': pattern_analysis,
        'model_analysis': model_analysis,
        'symbolic_results': symbolic_results,
        'parameter_results': parameter_results
    }

def main():
    """Main function to process circuits."""
    parser = argparse.ArgumentParser(description='Process circuit aging data')
    parser.add_argument('--fresh', type=str, required=True, help='Path to fresh simulation data CSV')
    parser.add_argument('--reduction', type=str, required=True, help='Path to leakage reduction data CSV')
    parser.add_argument('--circuit', type=str, required=True, help='Circuit name (e.g., NAND3, XOR2)')
    parser.add_argument('--output', type=str, default=None, help='Output directory for results')
    parser.add_argument('--no-train', action='store_true', help='Skip model training if models exist')
    
    args = parser.parse_args()
    
    # Process the circuit
    process_circuit(
        fresh_data_path=args.fresh,
        reduction_data_path=args.reduction,
        circuit_name=args.circuit,
        output_dir=args.output,
        train_models=not args.no_train
    )

if __name__ == "__main__":
    main()