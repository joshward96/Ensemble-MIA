# Ensembling Membership Inference Attacks Against Tabular Generative Models

The codebase for Ensembling Membership Inference Attacks Against Tabular Generative Models (https://arxiv.org/abs/2509.05350).

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/joshward96/Ensemble-MIA.git
cd ensemble
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Data Directory
```bash
mkdir -p data
# Place your CSV datasets in the data/ folder
```

## Quick Start

The evaluation follows a simple 3-step workflow:

### 1. Generate Synthetic Data (SynthCity Methods)
```bash
python generate_data_synthcity.py
```
Generates synthetic data using traditional methods: DDPM, ARF, TVAE, CTGAN, NFlow, ADSGAN, PATEGAN. (Note that this will take a very long time to generate data for the full experiment.)

### 2. Generate Latent Diffusion Data
```bash
python generate_data_latent_diff.py
```
Generates synthetic data using advanced latent diffusion: TabSyn, AutoDiff

### 3. Run MIA Evaluation
```bash
python run_mia_ensembles.py
```
Evaluates membership inference attacks using ensemble methods on all generated data.

## What It Does

- **Data Generation**: Creates synthetic versions of tabular datasets using 9 different methods
- **MIA Attacks**: Tests 11 different membership inference attacks (DCR, Gen-LRA, Classifier, etc.)
- **Ensemble Evaluation**: Combines attack results using aggregation and majority voting

## Output

Results are saved to `results/mia_results.csv` containing AUC-ROC, TPR@FPR metrics, and privacy scores for each dataset/method/attack combination.

## Synthetic Data Directory Structure

```
ensemble_data/
├── dataset_name/
│   ├── seed_X/
│   │   ├── mem_set.csv          # Training data
│   │   ├── holdout_set.csv      # Test data  
│   │   └── synth/
│   │       ├── ddpm/synth_1x.csv
│   │       ├── ctgan/synth_1x.csv
│   │       └── ...
```
