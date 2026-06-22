# src/plotting.py
from venv import logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
# from mpl_toolkits.mplot3d import Axes3D

class DataPlotter:
    """Class for generating exploratory data analysis plots."""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Initialize plotter with default settings.
        """
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_missing_values(self, missing_values, figsize=(10, 6), save=False, plot_dir: str = './results/plots/'):
        """Plot missing values per column."""
        plt.figure(figsize=figsize)
        missing_values.plot(kind="bar")
        plt.title("Missing Values in Each Column", fontsize=12, fontweight='bold')
        plt.xlabel("Columns")
        plt.ylabel("Number of Missing Values")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        if save:
            plt.savefig(f"{plot_dir}/missing_values.png", dpi=300, bbox_inches='tight')
            logger.info(f"Missing values plot saved to {plot_dir}/missing_values.png")
    
    def plot_class_distribution(self, counts, percentages, figsize=(6, 5), save=False, plot_dir: str = './results/plots/'):
        """Plot class distribution with percentages."""
        _ , ax = plt.subplots(figsize=figsize)
        bars = ax.bar(counts.index.astype(str), counts.values, color=["#1f77b4", "#ff7f0e"])
        ax.set_title("Class Distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel("Class Label")
        ax.set_ylabel("Count")
        ax.set_xticklabels(["Background (0)", "Signal (1)"])
        
        # Add percentages on bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, 
                f"{percentage}%\n{int(height):,}", 
                ha="center", va="bottom", fontsize=10)
        
        plt.tight_layout()
        plt.show()

        if save:
            plt.savefig(f"{plot_dir}/class_distribution.png", dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {plot_dir}/class_distribution.png")
    
    def plot_feature_variance(self, variance, threshold=0.01, figsize=(12, 5), save=False, plot_dir: str = './results/plots/'):
        """Plot feature variance with threshold line."""
        _ , (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Linear scale
        ax1.bar(range(len(variance)), variance, color='steelblue')
        ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold}')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Variance')
        ax1.set_title('Feature Variance (Linear Scale)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Log scale
        sorted_indices = np.argsort(variance)
        sorted_variance = variance[sorted_indices]
        ax2.semilogy(range(len(sorted_variance)), sorted_variance, 'o-', color='steelblue')
        ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold}')
        ax2.set_xlabel('Feature Index (sorted)')
        ax2.set_ylabel('Variance (log scale)')
        ax2.set_title('Feature Variance (Log Scale)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.show()

        if save:
            plt.savefig(f"{plot_dir}/feature_variance.png", dpi=300, bbox_inches='tight')
            logger.info(f"Feature variance plots saved to {plot_dir}/feature_variance.png")
    
    def plot_feature_distributions_by_class(self, X, y, n_features=20, figsize=(16, 10), save=False, plot_dir: str = './results/plots/'):
        """
        Plot distributions of features separated by class.
        """
        n_rows = (n_features + 4) // 5  # Dynamic grid
        _, axes = plt.subplots(n_rows, 5, figsize=figsize)
        axes = axes.flatten()

        for i in range(min(n_features, X.shape[1])):
            axes[i].hist(X[y == 0, i], bins=50, alpha=0.6, label='Background', density=True)
            axes[i].hist(X[y == 1, i], bins=50, alpha=0.6, label='Signal', density=True)
            axes[i].set_title(f'Feature {i+1}', fontsize=9)
            axes[i].grid(alpha=0.3)
            if i == 0:
                axes[i].legend(fontsize=8)

        plt.suptitle('Feature Distributions: Signal vs Background', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        if save:
            plt.savefig(f"{plot_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
            logger.info(f"Feature distribution plots saved to {plot_dir}/feature_distributions.png")
    
    def plot_correlation_matrix(self, X, feature_range=None, cmap='coolwarm', figsize=(10, 8), save=False, plot_dir: str = './results/plots/'):
        """
        Plot correlation matrix heatmap.
        """
        if feature_range is None:
            X_subset = X[:, :21]  # Default: low-level features
        else:
            start, end = feature_range
            X_subset = X[:, start:end]
        
        corr_matrix = np.corrcoef(X_subset.T)

        _ , ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, cmap=cmap, center=0, square=True, 
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix: Low-Level Features', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save:
            plt.savefig(f"{plot_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {plot_dir}/correlation_matrix.png")
        plt.show()
    
    def plot_class_separation(self, X, y, figsize=(12, 5), save=False, plot_dir: str = './results/plots/'):
        """Plot class separation for low-level vs high-level features."""
        X_signal = X[y == 1]
        X_background = X[y == 0]

        def separation(X_signal, X_background, feature_idx):
            mean_signal = X_signal[:, feature_idx].mean()
            mean_background = X_background[:, feature_idx].mean()
            return abs(mean_signal - mean_background)

        low_level_sep = [separation(X_signal, X_background, i) for i in range(21)]
        high_level_sep = [separation(X_signal, X_background, i) for i in range(21, 28)]

        _ , (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.bar(range(len(low_level_sep)), low_level_sep, color='steelblue')
        ax1.set_title('Low-Level Features: Class Separation', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Mean Separation')
        ax1.grid(alpha=0.3)

        ax2.bar(range(len(high_level_sep)), high_level_sep, color='coral')
        ax2.set_title('High-Level Features: Class Separation', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Mean Separation')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        if save:
            plt.savefig(f"{plot_dir}/class_separation.png", dpi=300, bbox_inches='tight')
            logger.info(f"Class separation plot saved to {plot_dir}/class_separation.png")
    
    def plot_pca_projection(self, X_scaled, y, n_components=2, figsize=(10, 7), save=False, plot_dir: str = './results/plots/'):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        if n_components == 2:
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], 
                    alpha=0.3, s=10, label='Background', c='blue')
            ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], 
                    alpha=0.6, s=20, label='Signal', c='red')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
            ax.set_title('2D PCA Projection: Signal vs Background', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            if save:
                plt.savefig(f"{plot_dir}/pca_projection.png", dpi=300, bbox_inches='tight')
                logger.info(f"PCA projection saved to {plot_dir}/pca_projection.png")
            plt.show()
        
        elif n_components == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], X_pca[y == 0, 2],
                    alpha=0.3, s=10, label='Background', c='blue')
            ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], X_pca[y == 1, 2],
                    alpha=0.6, s=20, label='Signal', c='red')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            ax.set_title('3D PCA Projection', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            plt.tight_layout()
            plt.show()

            if save:
                plt.savefig(f"{plot_dir}/pca_projection_3d.png", dpi=300, bbox_inches='tight')
                logger.info(f"3D PCA projection saved to {plot_dir}/pca_projection_3d.png")
    
    def plot_boxplot_features(self, X, y, feature_indices=None, figsize=(16, 5), save=False, plot_dir: str = './results/plots/'):
        if feature_indices is None:
            feature_indices = np.random.choice(X.shape[1], 5, replace=False)
        
        _ , axes = plt.subplots(1, len(feature_indices), figsize=figsize)
        
        for i, feature_idx in enumerate(feature_indices):
            axes[i].boxplot([X[y == 0, feature_idx], X[y == 1, feature_idx]], 
                            label=['Background', 'Signal'])
            axes[i].set_title(f'Feature {feature_idx+1}', fontsize=10)
            axes[i].grid(alpha=0.3)
        
        plt.suptitle('Feature Distributions (Box Plots)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save:
            plt.savefig(f"{plot_dir}/feature_boxplots.png", dpi=300, bbox_inches='tight')
            logger.info(f"Feature box plots saved to {plot_dir}/feature_boxplots.png")
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_probs, model_name="Model", ax=None, save=False, plot_dir: str = './results/plots/'):
        """Plot ROC curve."""
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        if ax is None:
            plt.tight_layout()
            plt.show()

        if save:
            plt.savefig(f"{plot_dir}/{model_name}_roc_curve.png", dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {plot_dir}/{model_name}_roc_curve.png")
        
        return roc_auc


    def plot_loss_history(self, loss_history, figsize=(12, 6), save=False, plot_dir: str = './results/plots/'):
        plt.figure(figsize=figsize)
        plt.plot(loss_history['train_loss'], label='Train Loss')
        plt.plot(loss_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(f"{plot_dir}/loss_history.png", dpi=300, bbox_inches='tight')
            logger.info(f"Loss history plot saved to {plot_dir}/loss_history.png")
        plt.show()


    def plot_roc_auc_history(self, loss_history, figsize=(12, 6), save=False, plot_dir: str = './results/plots/'):
        plt.figure(figsize=figsize)
        plt.plot(loss_history['val_roc_auc'], label='Validation ROC AUC', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('ROC AUC')
        plt.title('Validation ROC AUC Over Epochs')
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(f"{plot_dir}/roc_auc_history.png", dpi=300, bbox_inches='tight')
            logger.info(f"ROC AUC history plot saved to {plot_dir}/roc_auc_history.png")
        plt.show()

    def plot_brier_score_history(self, loss_history, figsize=(12, 6), save=False, plot_dir: str = './results/plots/'):
        plt.figure(figsize=figsize)
        plt.plot(loss_history['val_brier_score'], label='Validation Brier Score', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Brier Score')
        plt.title('Validation Brier Score Over Epochs')
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(f"{plot_dir}/brier_score_history.png", dpi=300, bbox_inches='tight')
            logger.info(f"Brier score history plot saved to {plot_dir}/brier_score_history.png")
        plt.show()