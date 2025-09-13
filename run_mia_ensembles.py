#!/usr/bin/env python3
"""
Refactored MIA evaluation script for generating comprehensive MIA results.

This script has been organized into separate utility modules for better maintainability:
- data_utils.py: Data preprocessing and file loading utilities
- evaluator_utils.py: MIA evaluation and ensemble methods  
- processing_utils.py: Main processing functions and configuration constants

Usage:
    python christy_eval_gen_mia_refactored.py
"""

import os
from utils import (
    process_and_evaluate, 
    save_mia_results_to_csv,
    DATASETS, 
    SEEDS, 
    METHODS
)


def main():
    """Main execution function"""
    # Configuration
    base_directory = "ensemble_data/"  # Replace with the correct base path
    output_file = 'results/mia_results.csv'
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Process all combinations of datasets, seeds, and methods
    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}")
        for seed in SEEDS:
            print(f"Using seed: {seed}")
            for method in METHODS:
                print(f"Using method: {method}")
                try:
                    results = process_and_evaluate(
                        base_directory, dataset, seed, method
                    )
                    save_mia_results_to_csv(*results, output_file=output_file)
                    print(f"Successfully processed {dataset} with {method}")
                except Exception as e:
                    print(f"Error processing {dataset} with {method}: {str(e)}")


if __name__ == "__main__":
    main()
