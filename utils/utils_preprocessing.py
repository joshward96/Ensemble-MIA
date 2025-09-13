"""
Main processing and result saving utilities for MIA evaluation
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils_data import load_required_files, ordinal_tabular_preprocess
from utils_evaluator import MIAEvaluator


def process_and_evaluate(base_path, dataset, seed, method):
    """
    Loads data, preprocesses it, and runs MIA evaluation using MIAEvaluator.
    Modified to work without n_size parameter.

    Args:
        base_path (str): The root directory path to start searching from.
        dataset (str): The dataset name.
        seed (str): The seed folder name.
        method (str): The method folder name.
    Returns:
        tuple: (dataset, seed, method, results)
    """
    base_path = Path(base_path)

    print(f"\nProcessing dataset: {dataset} with method: {method}")
    print("Loading required files...")
    dfs = load_required_files(base_path, dataset, seed, method)
    
    print("Preprocessing data...")
    mem, non_mem, synth, ref, transformer = ordinal_tabular_preprocess(
        dfs['mem'], dfs['non_mem'], dfs['synth'], dfs['ref']
    )

    try:
        print("Running MIA evaluation...")
        evaluator = MIAEvaluator()
        results = evaluator.evaluate(mem, non_mem, synth, ref)
        return dataset, seed, method, results
    except Exception as e:
        print(f"Error occurred during MIA evaluation: {str(e)}")
        return dataset, seed, method, None


def save_mia_results_to_csv(dataset, seed, method, results, output_file='mia_comprehensive_results_gen_mia.csv'):
    """
    Saves the MIA evaluation results to a CSV file.
    Modified to work without n_size parameter.

    Args:
        dataset (str): The dataset name.
        seed (str): The seed folder name.
        method (str): The method used.
        results (dict): The MIA evaluation results.
        output_file (str): The output CSV file name.
    """
    if results is None:
        print(f"No results to save for {dataset} with {method}")
        return
    
    all_rows = []
    
    for attack_method, metrics in results.items():
        row = {
            'dataset': dataset,
            'seed': seed,
            'method': method,
            'attack_method': attack_method,
            'auc_roc': metrics.get('auc_roc', np.nan),
            'tpr_at_fpr_0': metrics.get('tpr_at_fpr_0', np.nan),
            'tpr_at_fpr_0.001': metrics.get('tpr_at_fpr_0.001', np.nan),
            'tpr_at_fpr_0.01': metrics.get('tpr_at_fpr_0.01', np.nan),
            'tpr_at_fpr_0.1': metrics.get('tpr_at_fpr_0.1', np.nan)
        }
        all_rows.append(row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_rows)
    
    # Append to CSV file
    results_df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)
    print(f"Saved results for {dataset} with {method} to {output_file}")


# Configuration constants
DATASETS = [
    'analcatdata_authorship',
    'banknote-authentication',
    'blood-transfusion-service-center',
    'breast-w',
    'car',
    'climate-model-simulation-crashes',
    'credit-g',
    'cylinder-bands',
    'jm1',
    'mfeat-fourier',
    'mfeat-karhunen',
    'mfeat-morphological',
    'mfeat-zernike',
    'ozone-level-8hr',
    'pc1',
    'pendigits',
    'phoneme',
    'segment',
    'spambase',
    'splice',
    'analcatdata_dmft',
    'balance-scale',
    'churn',
    'cmc',
    'chae-9',
    'connect-4',
    'credit-approval',
    'diabetes',
    'dna',
    'dresses-sales',
    'eucalyptus',
    'first-order-theorem-proving',
    'ilpd',
    'Internet-Advertisements',
    'kc1',
    'kr-vs-kp',
    'mfeat-pixel',
    'MiceProtein',
    'optdigits',
    'pc3',
    'pc4',
    'qsar-biodeg',
    'satimage',
    'sick',
    'steel-plates-fault',
    'texture',
    'tic-tac-toe',
    'vehicle',
    'vowel',
    'wall-robot-navigation',
    'wdbc',
    'wilt'
]

SEEDS = ['seed_1', 'seed_2', 'seed_3', 'seed_4', 'seed_5']

METHODS = ['adsgan', 'arf', 'ctgan', 'ddpm', 'nflow', 'pategan', 'tvae', 'tabsyn', 'autodiff']
