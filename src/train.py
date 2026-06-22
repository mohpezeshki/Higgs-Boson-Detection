from joblib import logger
from sklearn.metrics import roc_auc_score, brier_score_loss
from utils import save_model_checkpoint, load_model_checkpoint
from tqdm.auto import tqdm
import torch


from sklearn.metrics import roc_auc_score, brier_score_loss
from utils import save_model_checkpoint, load_model_checkpoint
from tqdm.auto import tqdm

# use accuracy_score for evaluation,
# because the dataset is balanced and we want a simple metric to track during training.

class HiggsTraining:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device


    def train_step(self, dataloader):
        self.model.train()
        total_loss = 0.0
        
        for inputs, labels in dataloader:    
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.squeeze())
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()        
        total_loss /= len(dataloader)

        return total_loss


    @torch.inference_mode()    
    def validate_step(self, dataloader):
    
        roc_auc = 0.0
        bsl = float('inf')
        loss = 0.0
        all_labels = []
        all_probs = []
        self.model.eval()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).argmax(dim=1).cpu().numpy())
            loss += self.criterion(outputs, labels.squeeze()).item()

        loss = loss / len(dataloader)
        roc_auc = roc_auc_score(all_labels, all_probs)
        bsl = brier_score_loss(all_labels, all_probs)
        return loss, roc_auc, bsl

    @torch.inference_mode()  
    def test_step(self, dataloader):
        self.model.eval()

        loss = 0.0
        roc_auc = 0.0
        bsl = float('inf')
        all_labels = []
        all_probs = []
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss += self.criterion(outputs, labels.squeeze())
            
            probs = torch.softmax(outputs, dim=1).argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        loss = loss / len(dataloader)
        roc_auc = roc_auc_score(all_labels, all_probs)
        bsl = brier_score_loss(all_labels, all_probs)
        return loss.item(), roc_auc, bsl, all_labels, all_probs  
    
    def fit(self, train_loader, val_loader, epochs, checkpoint_path=None):
        
        loss_history = {
            "train_loss": [],
            "val_loss": [],
            "val_roc_auc": [],
            "val_brier_score": []
        }

        max_roc_auc = 0.0

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            train_loss = self.train_step(train_loader)
            val_loss, val_roc_auc, val_brier_score = self.validate_step(val_loader)
            loss_history["train_loss"].append(train_loss)
            loss_history["val_loss"].append(val_loss)
            loss_history["val_roc_auc"].append(val_roc_auc)
            loss_history["val_brier_score"].append(val_brier_score)

            if val_roc_auc > max_roc_auc and checkpoint_path:
                max_roc_auc = val_roc_auc
                save_model_checkpoint(self.model, self.optimizer, epoch, loss_history, f"{checkpoint_path}/best_model.pt")
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val ROC AUC: {val_roc_auc:.4f}, Val Brier Score: {val_brier_score:.4f}")
        return loss_history