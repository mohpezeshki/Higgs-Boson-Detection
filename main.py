import pandas as pd
import Pathlib


import torch
from torch.utils.data import DataLoader
from torch import nn

from src.utils import create_random_sample
from src.data import HIGGSDataset
from src.models import create_nn_model, create_XGBoost_model
from src.train import HiggsTraining
from src.config import CONFIG
from src.evaluate import HiggsEvaluator

from logging import logging
logger = logging.getLogger(__name__)

def main(config=CONFIG):

    if config['dataset'] == 'higgs-uci-dataset':
        train_data_dir = config['train_data_dir'].format(dataset=config['dataset'])
        val_data_dir = config['val_data_dir'].format(dataset=config['dataset'])
        test_data_dir = config['test_data_dir'].format(dataset=config['dataset'])
    
    if train_data_dir is None or val_data_dir is None or test_data_dir is None:
        raise ValueError("Data directories are not properly set in the configuration.")
    
    if not Pathlib.Path(train_data_dir).exists() or not Pathlib.Path(val_data_dir).exists() or not Pathlib.Path(test_data_dir).exists():
        raise ValueError("One or more data directories do not exist.")

    X_train = pd.read_csv(f"{train_data_dir}/train_dataset.csv").values
    y_train = pd.read_csv(f"{train_data_dir}/train_labels.csv").values
    X_val = pd.read_csv(f"{val_data_dir}/val_dataset.csv").values
    y_val = pd.read_csv(f"{val_data_dir}/val_labels.csv").values
    X_test = pd.read_csv(f"{test_data_dir}/test_dataset.csv").values
    y_test = pd.read_csv(f"{test_data_dir}/test_labels.csv").values


    if config.get('use_random_sample', False):
        X_train, y_train = create_random_sample(X_train, y_train, sample_size=config.get('random_sample_size', 10000))
        X_val, y_val = create_random_sample(X_val, y_val, sample_size=config.get('random_sample_size', 10000))
        X_test, y_test = create_random_sample(X_test, y_test, sample_size=config.get('random_sample_size', 10000))


        
    train_dataset = HIGGSDataset(X_train, y_train)
    test_dataset = HIGGSDataset(X_test, y_test)
    val_dataset = HIGGSDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    higgs_model = create_nn_model(input_dim=28, config=config)
    hg_training = HiggsTraining(
        model=higgs_model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(higgs_model.parameters(), lr=config['learning_rate'], weight_decay=config['l2_reg']),
        device=config['device']
    )

    logger.info("Starting training NN model...")
    hg_training.fit(train_loader, val_loader, epochs=config['epochs'], checkpoint_path=f"{config['checkpoint_dir']}/higgs_model.pth")

    logger.info("Training NN model completed.")


    logger.info("Starting evaluation of NN model...")
    test_loss, test_roc_auc, test_brier_score, _ , _ = hg_training.test_step(test_loader)
    logger.info(f"NN Model Test Loss: {test_loss:.4f}, Test ROC AUC: {test_roc_auc:.4f}, Test Brier Score: {test_brier_score:.4f}")
    test_preds_nn_model = higgs_model.predict(X_test, device=config['device'])

    nn_mdoel_evaluator = HiggsEvaluator(y_test.squeeze(), test_preds_nn_model, "Neural Network")
    evaluation_results_nn = nn_mdoel_evaluator.evaluate()

    logger.info("NN Model Evaluation Results:")
    for metric, value in evaluation_results_nn.items():
        logger.info(f"{metric}: {value:.4f}")



    logger.info("Starting evaluation of XGBoost model...")
    xgboost_model = create_XGBoost_model()

    xgboost_model.fit(X_train, y_train)
    test_preds_xgb = xgboost_model.predict(X_test)

    logger.info("XGBoost Model Evaluation Results:")
    xgb_evaluator = HiggsEvaluator(y_test.squeeze(), test_preds_xgb, "XGBoost")
    evaluation_results_xgb = xgb_evaluator.evaluate()
    for metric, value in evaluation_results_xgb.items():
        logger.info(f"{metric}: {value:.4f}")

    



if __name__ == "__main__":
    main()
