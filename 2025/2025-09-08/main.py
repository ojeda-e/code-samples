"""
Main script demonstrating misleading accuracy with imbalanced data.

This script shows how accuracy can be misleading when dealing with imbalanced datasets,
and why metrics like F1-score, precision, and recall are more informative.
"""

import logging
import numpy as np
import torch
import random

from generate_data import generate_imbalanced_data
from build_model import train_model
from metrics import evaluate_model_performance, compare_models
from visualization import create_all_visualizations

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

SEED = 12345

def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility")

def main():
    """Main execution function demonstrating misleading accuracy."""
    set_seed(SEED)
    
    logger.info("Demonstrating Misleading Accuracy with Imbalanced Data")
    logger.info("=" * 60)
    
    # Step 1: Generate imbalanced data
    logger.info("\nStep 1: Generating imbalanced data...")
    df_data = generate_imbalanced_data(seed=SEED)
    
    # Show data distribution
    positive_ratio = df_data['binds_target'].mean()
    logger.info("Data Distribution:")
    logger.info(f"  Total samples: {len(df_data):,}")
    logger.info(f"  Positive class: {positive_ratio:.1%} ({df_data['binds_target'].sum():,} samples)")
    logger.info(f"  Negative class: {1-positive_ratio:.1%} ({(1-df_data['binds_target']).sum():,} samples)")
    
    # Step 2: Train Random Forest model
    logger.info("\nStep 2: Training Random Forest model...")
    rf_results = train_model(df_data, model_type="random_forest")
    
    # Step 3: Train Neural Network model
    logger.info("\nStep 3: Training Neural Network model...")
    nn_results = train_model(df_data, model_type="neural_network")
    
    # Step 4: Evaluate models with standard 0.5 threshold
    logger.info("\n" + "="*60)
    logger.info("Step 4: Evaluating models with standard 0.5 threshold")
    logger.info("="*60)
    
    # Evaluate Random Forest
    rf_predictions = rf_results['predictions']['test']
    if 'final_preds' in rf_predictions and 'final_targets' in rf_predictions:
        rf_y_true = rf_predictions['final_targets']
        rf_y_proba = rf_predictions['final_preds']
        rf_y_pred = (rf_y_proba > 0.5).astype(int)  # Standard threshold
        
        logger.info("\n ➡️ Random Forest Model:")
        rf_metrics =evaluate_model_performance(rf_y_true, rf_y_pred, rf_y_proba, "Random Forest")
    
    # Evaluate Neural Network
    nn_predictions = nn_results['predictions']['test']
    if 'final_preds' in nn_predictions and 'final_targets' in nn_predictions:
        nn_y_true = nn_predictions['final_targets']
        nn_y_proba = nn_predictions['final_preds']
        nn_y_pred = (nn_y_proba > 0.5).astype(int)  # Standard threshold
        
        logger.info("\n ➡️ Neural Network Model:")
        nn_metrics = evaluate_model_performance(nn_y_true, nn_y_pred, nn_y_proba, "Neural Network")
    
    compare_models(rf_metrics, nn_metrics)
    
    # Generate plots
    logger.info("\n" + "="*60)
    logger.info("Step 5: Creating visualizations")
    logger.info("="*60)
    
    create_all_visualizations(
        df_data, rf_metrics, nn_metrics,
        rf_y_true, rf_y_pred, rf_y_proba,
        nn_y_true, nn_y_pred, nn_y_proba,
        save_dir="/tmp/plots"
    )

if __name__ == "__main__":
    main()