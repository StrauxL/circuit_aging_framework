# process_new_circuit.py

import os
import argparse
import logging
from src.main import process_circuit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to process a new circuit."""
    parser = argparse.ArgumentParser(description='Process a new circuit')
    parser.add_argument('--fresh', type=str, required=True, help='Path to fresh simulation data CSV')
    parser.add_argument('--reduction', type=str, required=True, help='Path to leakage reduction data CSV')
    parser.add_argument('--circuit', type=str, required=True, help='Circuit name (e.g., XOR2, NAND2)')
    parser.add_argument('--output', type=str, default=None, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.fresh):
        raise FileNotFoundError(f"Fresh simulation data file not found: {args.fresh}")
    
    if not os.path.exists(args.reduction):
        raise FileNotFoundError(f"Reduction data file not found: {args.reduction}")
    
    # Set up output directory
    if args.output is None:
        args.output = f"./results/{args.circuit}"
    
    os.makedirs(args.output, exist_ok=True)
    
    logger.info(f"Processing new circuit: {args.circuit}")
    logger.info(f"Fresh data: {args.fresh}")
    logger.info(f"Reduction data: {args.reduction}")
    logger.info(f"Output directory: {args.output}")
    
    # Process the circuit
    process_circuit(
        fresh_data_path=args.fresh,
        reduction_data_path=args.reduction,
        circuit_name=args.circuit,
        output_dir=args.output,
        train_models=True
    )
    
    logger.info(f"Circuit {args.circuit} processing completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()