from torch.utils.data import Dataset
from pyprojroot import here
import torch
import numpy as np

class HIGGSDataset(Dataset):
    def __init__(self, X, y,  class_labels=["background", "signal"]):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.num_classes = len(class_labels)
        self.class_labels = class_labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]
        return features, label
    
    def label_to_class(self, label):
        return self.class_labels[int(label)]
    
    def __repr__(self):
        return f"HIGGSDataset(num_samples={len(self)}, num_classes={self.num_classes})"
    

    def __str__(self):
        return f"HIGGSDataset(num_samples={len(self)}, num_classes={self.num_classes})"