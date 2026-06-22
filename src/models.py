from torch import nn
from typing import List
import torch


class HiggsModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims: List[int]):
        super(HiggsModel, self).__init__()
        fc1 = nn.Linear(input_dim, hidden_dims[0])
        layers = [fc1]
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))

        last_fc = nn.Linear(hidden_dims[-1], output_dim)
        layers.append(last_fc)

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
    
    def predict(self, X , device='cpu'):
        self.eval()
        with torch.inference_mode():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            outputs = self.forward(inputs)
            probs = torch.softmax(outputs, dim=1).argmax(dim=1)
        return probs.cpu().numpy() 
    

def create_nn_model(input_dim, config):
    model = HiggsModel(input_dim=input_dim, output_dim=config['output_dim'], hidden_dims=config['hidden_sizes'])
    model = model.to(config['device'])
    return model

def create_XGBoost_model(input_dim, output_dim, hidden_dims: List[int]):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    return model
    

