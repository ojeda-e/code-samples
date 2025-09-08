"""
Model evaluation and metrics utilities.

This module contains code for evaluating model performance and printing metrics using torchmetrics.

Notes
----------
- torchmetrics is used for calculating metrics.
- sklearn.metrics is used for printing classification report.
"""

import logging
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision, ConfusionMatrix
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

METRICS = [
    ("Accuracy", "accuracy"),
    ("AUROC", "auroc"),
    ("PR-AUC", "pr_auc"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1 Score", "f1"),
]

KEY_MAP = {
    "Accuracy": "accuracy",
    "AUROC": "auroc",
    "PR-AUC": "pr_auc",
    "Precision": "precision",
    "Recall": "recall",
    "F1 Score": "f1_score",
}

def evaluate_model_performance(y_true, y_pred, y_pred_proba, model_name: str):
    """Evaluate model performance and print metrics using torchmetrics."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating {model_name} Model")
    logger.info(f"{'='*50}")
    
    # Convert to torch tensors
    y_true_tensor = torch.tensor(y_true, dtype=torch.long)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.long)
    y_pred_proba_tensor = torch.tensor(y_pred_proba, dtype=torch.float32)
    
    # Ensure predictions are 1D (squeeze if needed)
    if y_pred_tensor.dim() > 1:
        y_pred_tensor = y_pred_tensor.squeeze()
    if y_pred_proba_tensor.dim() > 1:
        y_pred_proba_tensor = y_pred_proba_tensor.squeeze()
    
   
    # Initialize torchmetrics for binary classification and calculate metrics
    accuracy_metric = Accuracy(task="binary")
    precision_metric = Precision(task="binary")
    recall_metric = Recall(task="binary")
    f1_metric = F1Score(task="binary")
    auroc_metric = AUROC(task="binary")
    pr_auc_metric = AveragePrecision(task="binary")
    cm_metric = ConfusionMatrix(task="binary", num_classes=2)
    

    accuracy = accuracy_metric(y_pred_tensor, y_true_tensor).item()
    precision = precision_metric(y_pred_tensor, y_true_tensor).item()
    recall = recall_metric(y_pred_tensor, y_true_tensor).item()
    f1 = f1_metric(y_pred_tensor, y_true_tensor).item()
    auroc = auroc_metric(y_pred_proba_tensor, y_true_tensor).item()
    pr_auc = pr_auc_metric(y_pred_proba_tensor, y_true_tensor).item()
    cm = cm_metric(y_pred_tensor, y_true_tensor)
    
    # Print everything
    logger.info(f"\nClassification Report for {model_name}:")
    logger.info(classification_report(y_true, y_pred, zero_division=0))
    
    logger.info(f"\nConfusion Matrix for {model_name}:")
    cm_np = cm.numpy()
    logger.info(f"True Negatives: {cm_np[0,0]}, False Positives: {cm_np[0,1]}")
    logger.info(f"False Negatives: {cm_np[1,0]}, True Positives: {cm_np[1,1]}")

    logger.info(f"\n{model_name} Metrics:")
    
    metrics_dict = {name: locals()[key] for name, key in METRICS}
    
    for metric_name, value in metrics_dict.items():
        logger.info(f"{metric_name}: {value:.4f}")
    

    return {
        'accuracy': accuracy,
        'auroc': auroc,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm_np.tolist()
    }

def compare_models(rf_metrics, nn_metrics):
    """Compare model performance metrics between Random Forest and Neural Network"""
    logger.info("\n ✨ ✨ ✨ Model Comparison: \n")
    header = f"{'Metric':<12} {'Random Forest':<15} {'Neural Network':<15}"
    logger.info(header)
    logger.info("-" * len(header))

    
    for name, _ in METRICS:
        key = KEY_MAP[name]
        logger.info(f"{name:<12} {rf_metrics[key]:<15.4f} {nn_metrics[key]:<15.4f}")