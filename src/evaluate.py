from tqdm.asyncio import tqdm

from plotting import DataPlotter
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, 
    confusion_matrix, precision_score, recall_score, 
    f1_score, accuracy_score, matthews_corrcoef
)
import numpy as np
import logging 
logger = logging.getLogger(__name__)


import matplotlib.pyplot as plt

class HiggsEvaluator:
    """Compute all metrics and generate plots"""
    
    def __init__(self, y_true, y_pred_probs, model_name):
        self.y_true = y_true
        self.y_pred = y_pred_probs
        self.model_name = model_name

        self.plotter = DataPlotter()  
    
    def evaluate(self):
        """Compute all evaluation metrics"""
        
        # Basic metrics
        auc = roc_auc_score(self.y_true, self.y_pred)
        threshold, ams = self.find_optimal_ams_threshold()[:2]
        
        # Get binary predictions at optimal threshold
        y_pred_binary = (self.y_pred >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred_binary).ravel()
        
        # Probability-based metrics
        brier = brier_score_loss(self.y_true, self.y_pred)
        logloss = log_loss(self.y_true, self.y_pred)
        
        # Classification metrics at optimal threshold
        accuracy = accuracy_score(self.y_true, y_pred_binary)
        precision = precision_score(self.y_true, y_pred_binary, zero_division=0)
        recall = recall_score(self.y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(self.y_true, y_pred_binary, zero_division=0)
        mcc = matthews_corrcoef(self.y_true, y_pred_binary)
        
        # Derived metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = recall  # Same as recall
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        
        # Return comprehensive metrics dictionary
        return {
            'model': self.model_name,
            'auc': float(auc),
            'brier_score': float(brier),
            'log_loss': float(logloss),
            'optimal_threshold': float(threshold),
            'ams': float(ams),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'matthews_corrcoef': float(mcc),
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr),
            'total_samples': len(self.y_true),
        }
    

    def save_evaluation_results(self, results_dict, filepath):
        """Save evaluation results to a JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=4)
        logger.info(f"Saved evaluation results to {filepath}")
    
    def compute_ams(self, threshold: float = 0.5) -> float:
        """
        Compute AMS (Approximate Median Significance).
        
        AMS = sqrt(2 * [s * ln(1 + s/b) - s])
        where:
            s = true positives (signal correctly classified)
            b = false positives (background misclassified)
        """
        y_pred = (self.y_pred >= threshold).astype(int)
        s = np.sum(np.logical_and(y_pred == 1, self.y_true == 1)) # TP
        b = np.sum(np.logical_and(y_pred == 1, self.y_true == 0))  # FP

        if b <= 0:
            return np.nan
        ams = np.sqrt(np.abs(2 * (s * np.log(1 + s / b) - s)))

        logger.debug(f"{self.model_name} AMS at threshold {threshold:.3f}: s={s}, b={b}, AMS={ams:.4f}")
        return ams
    
    def find_optimal_ams_threshold(self, n_thresholds: int = 1001) -> tuple:
        """
        Find threshold that maximizes AMS.
        
        Returns:
            (optimal_threshold, max_ams, all_ams_scores)
        """
        thresholds = np.linspace(0, 1, n_thresholds) 

        ams_scores = list(map(self.compute_ams, thresholds))
        
        optimal_idx = np.nanargmax(ams_scores)
        optimal_threshold = thresholds[optimal_idx]
        max_ams = ams_scores[optimal_idx]
        
        logger.info(f"{self.model_name} Optimal AMS: {max_ams:.4f} "
                f"at threshold {optimal_threshold:.3f}")
        
        return optimal_threshold, max_ams, np.array(ams_scores)

    def plot_roc_curve(self, ax):
        """ROC curve (TPR vs FPR)"""
        self.plotter.plot_roc_curve(self.y_true, self.y_pred, ax=ax, label=self.model_name)
    
    def plot_ams_curve(self, ax):
        """AMS vs Threshold"""
        thresholds = np.linspace(0, 1, 100)
        ams_scores = list(map(self.compute_ams, thresholds))
        ax.plot(thresholds, ams_scores, label=self.model_name)
        ax.set_xlabel('Decision Threshold')
        ax.set_ylabel('AMS Score')
        ax.legend()