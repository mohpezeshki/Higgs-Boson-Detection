
import torch

CONFIG = {

    'use_random_sample': True,
    'random_sample_size': 10000,

    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    'test_size': 0.1,  
    'val_size': 0.1, 
    
    # Model architecture
    'hidden_sizes': [500, 300, 200, 100, 50],
    'dropout_rate': 0.2,
    'output_dim': 2,
    'input_dim': 28,
    
    # Training
    'batch_size': 1024,  
    'learning_rate': 1e-3,
    'l2_reg': 1e-6,       # Weight decay (weak L2)
    'epochs': 100,
    'early_stopping_patience': 15,
    
    # Data paths
    'dataset' : 'higgs-uci-dataset',
    'data_dir': '../data/raw/{dataset}',
    'train_data_dir': '../data/processed/{dataset}/train/',
    'val_data_dir': '../data/processed/{dataset}/val/',
    'test_data_dir': '../data/processed/{dataset}/test/',
    'checkpoint_dir': '../results/models/',
    'plot_dir': './results/plots/',
}