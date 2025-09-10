"""
Visualization utilities for imbalanced data analysis.

This module contains functions to create visualizations that help understand
model performance on imbalanced datasets, including confusion matrices,
ROC curves, precision-recall curves, and threshold analysis.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Optional

logger = logging.getLogger(__name__)


plt.style.use('default')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)

# Custom palette to match my website
custom_palette = ["#31708E", "#5298AD", "#82B3C9", "#B0D0DC", "#DCEBF1"]
sns.set_palette(custom_palette)

def plot_class_distribution(y_true: np.ndarray, title: str = "Class Distribution", 
                          save_path: Optional[str] = None) -> None:
    """
    Plot the class distribution to show data imbalance.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels (0 and 1)
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    unique, counts = np.unique(y_true, return_counts=True)
    class_names = ['Inactive', 'Active']
    colors = ['#82B3C9', '#31708E']
    
    bars = ax1.bar(class_names, counts, color=colors, alpha=0.7, edgecolor='black', width=0.4)
    ax1.set_title(f'{title} - Counts', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    wedges, texts, autotexts = ax2.pie(counts, labels=class_names, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'{title} - Proportions', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1000)
    
    # Format percentage text
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str, save_path: Optional[str] = None) -> None:
    """
    Plot a confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    model_name : str
        Name of the model for the title
    save_path : str, optional
        Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    # Use cubehelix colormap - matches my website
    cubehelix_cmap = sns.cubehelix_palette(start=.5, rot=-.5, dark=0.5, light=.95, as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cubehelix_cmap, 
                xticklabels=['Predicted Inactive', 'Predicted Active'],
                yticklabels=['Actual Inactive', 'Actual Active'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    
    # Add performance summary
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    summary_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}'
    plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()

def plot_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray, 
                      model_name: str, save_path: Optional[str] = None) -> None:
    """
    Plot ROC and Precision-Recall curves side by side.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    model_name : str
        Name of the model for the title
    save_path : str, optional
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = np.trapz(tpr, fpr)
    
    ax1.plot(fpr, tpr, color='#31708E', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='#82B3C9', lw=2, linestyle='--', label='Random classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve - {model_name}', fontweight='bold')
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = np.trapz(precision, recall)
    
    ax2.plot(recall, precision, color='#31708E', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Add baseline (random classifier performance)
    baseline = np.sum(y_true) / len(y_true)
    ax2.axhline(y=baseline, color='#82B3C9', linestyle='--', 
                label=f'Random classifier (baseline = {baseline:.3f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve - {model_name}', fontweight='bold')
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC/PR curves plot saved to {save_path}")
    
    plt.show()

def plot_metrics_comparison(rf_metrics: dict, nn_metrics: dict, 
                           save_path: Optional[str] = None) -> None:
    """
    Plot a comparison of metrics between two models.
    
    Parameters
    ----------
    rf_metrics : dict
        Random Forest metrics
    nn_metrics : dict
        Neural Network metrics
    save_path : str, optional
        Path to save the plot
    """
    metrics = ['accuracy', 'roc_auc', 'pr_auc', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1 Score']
    
    rf_values = [rf_metrics[metric] for metric in metrics]
    nn_values = [nn_metrics[metric] for metric in metrics]
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, rf_values, width, label='Random Forest', 
                   color='#82B3C9', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, nn_values, width, label='Neural Network', 
                   color='#31708E', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison plot saved to {save_path}")
    
    plt.show()

def create_all_visualizations(df, rf_metrics, nn_metrics, rf_y_true, rf_y_pred, rf_y_proba, 
                             nn_y_true, nn_y_pred, nn_y_proba, save_dir: str = "plots") -> None:
    """
    Create all visualizations for the blog post.
    
    Parameters
    ----------
    df : polars.DataFrame
        Dataset with 'binds_target' column
    rf_metrics : dict
        Random Forest metrics
    nn_metrics : dict
        Neural Network metrics
    rf_y_true, rf_y_pred, rf_y_proba : np.ndarray
        Random Forest predictions
    nn_y_true, nn_y_pred, nn_y_proba : np.ndarray
        Neural Network predictions
    save_dir : str
        Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
       
    # Fig 1 Class distribution
    if not hasattr(df['binds_target'], 'values'):
        y_true = df['binds_target'].to_numpy()
    
    plot_class_distribution(y_true, 
                           save_path=f"{save_dir}/class_distribution.png")
    
    # Fig 2 and 3 Confusion matrices
    plot_confusion_matrix(rf_y_true, rf_y_pred, "Random Forest", 
                         save_path=f"{save_dir}/confusion_matrix_rf.png")
    plot_confusion_matrix(nn_y_true, nn_y_pred, "Neural Network", 
                         save_path=f"{save_dir}/confusion_matrix_nn.png")
    
    # Fig 4 Metrics comparison
    plot_metrics_comparison(rf_metrics, nn_metrics, 
                           save_path=f"{save_dir}/metrics_comparison.png")

    # Fig 5 and 6 ROC and PR curves
    plot_roc_pr_curves(rf_y_true, rf_y_proba, "Random Forest", 
                      save_path=f"{save_dir}/roc_pr_curves_rf.png")
    plot_roc_pr_curves(nn_y_true, nn_y_proba, "Neural Network", 
                      save_path=f"{save_dir}/roc_pr_curves_nn.png")




