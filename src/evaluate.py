from tqdm.asyncio import tqdm

from plotting import DataPlotter
from sklearn.metrics import roc_auc_score, roc_curve
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
        """Compute all metrics"""
        auc = roc_auc_score(self.y_true, self.y_pred)
        threshold, ams = self.find_optimal_ams_threshold()[:2]
        
        logger.info(f"{self.model_name} AUC: {auc:.4f}, Optimal AMS: {ams:.4f} at threshold {threshold:.3f}")
        return {
            'model': self.model_name,
            'auc': auc,
            'optimal_threshold': threshold,
            'ams_at_optimal': ams,
        }
    
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

def compare_models(results: dict, plot_dir: str = './results/plots/'):
    """
    Create comparison plots for multiple models.
    
    Args:
        results: Dict mapping model_name -> {'evaluator': HiggsEvaluator}
        plot_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for model_name, data in results.items():
        evaluator = data['evaluator']
        evaluator.plot_roc_curve(axes[0])
        evaluator.plot_ams_curve(axes[1])
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {plot_dir}/model_comparison.png")
    plt.show()