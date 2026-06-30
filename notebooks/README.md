# Notebooks

Exploration, training, and evaluation workflow for the Higgs boson signal/background classification task. Run in order — each notebook depends on artifacts produced by the previous one.

## 01_data_exploration.ipynb

Initial look at the HIGGS dataset (see [data/](../data)).

- Load data & basic info
- Check for missing values
- Class distribution
- Feature variance and distribution analysis by class
- Correlation between features
- Low-level vs. high-level feature comparison
- PCA visualization (2D projection)
- Box plots (quartiles & outliers)
- Summary & key findings

## 02_training.ipynb

Trains both models defined in [src/models.py](../src/models.py) on the prepared train/val/test splits.

- Builds the `HiggsModel` feedforward neural network (PyTorch) and an XGBoost baseline
- Configures hyperparameters (architecture, learning rate, regularization, epochs, device)
- Samples training/validation/test subsets and wraps them in `HIGGSDataset` / `DataLoader`
- Trains the NN via `HiggsTraining.fit`, with loss/ROC-AUC/Brier score curves plotted over epochs
- Saves checkpoints and generates predictions for both models on the test set

## 03_results_analysis.ipynb

Compares the trained Neural Network and XGBoost models using the evaluation results written to [results/evaluation/](../results/evaluation).

- Loads `nn_model_evaluation_results.json` and `xgb_evaluation_results.json`
- Builds a side-by-side metrics summary (AUC, Accuracy, Precision, Recall, F1, AMS, Brier, Log Loss)
- Plots metric comparisons, confusion matrices, error rates (FPR/FNR), and AMS-vs-threshold curves
- Figures are saved to [results/plots/](../results/plots) and referenced in the top-level [README](../README.md#results)

## Setup

Run from the project root with the project's virtual environment active (see [pyproject.toml](../pyproject.toml) / [requirements.txt](../requirements.txt)):

```bash
jupyter lab notebooks/
```

Data must be downloaded and prepared first via [scripts/download_data.py](../scripts/download_data.py) and [scripts/prepare_data.py](../scripts/prepare_data.py).
