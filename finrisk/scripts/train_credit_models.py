#!/usr/bin/env python3
"""
Credit Risk Model Training Script for FinRisk Application.

This script demonstrates how to train credit risk models using the comprehensive
training module. It includes data preparation, model training, evaluation, and
model interpretability features.

Usage:
    python scripts/train_credit_models.py [--no-tuning] [--test-size 0.2]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.models.credit_risk_trainer import (
    CreditRiskModelTrainer, 
    CreditModelExplainer, 
    train_credit_models
)
from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('credit_model_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train credit risk models')
    parser.add_argument(
        '--no-tuning', 
        action='store_true', 
        help='Skip hyperparameter tuning'
    )
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2, 
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--explain', 
        action='store_true', 
        help='Generate model explanations'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting credit risk model training...")
        logger.info(f"Hyperparameter tuning: {not args.no_tuning}")
        logger.info(f"Test size: {args.test_size}")
        
        # Train models
        results = train_credit_models(hyperparameter_tuning=not args.no_tuning)
        
        # Display results
        print("\n" + "="*80)
        print("CREDIT RISK MODEL TRAINING RESULTS")
        print("="*80)
        
        print("\nModel Performance Comparison:")
        print(results['comparison'])
        
        # Find best model
        best_model_name = results['comparison']['auc'].idxmax()
        best_auc = results['comparison'].loc[best_model_name, 'auc']
        print(f"\nBest Model: {best_model_name} (AUC: {best_auc:.4f})")
        
        # Model explanations if requested
        if args.explain and best_model_name in results['models']:
            print("\n" + "="*80)
            print("MODEL EXPLANATIONS")
            print("="*80)
            
            best_model_info = results['models'][best_model_name]
            explainer = CreditModelExplainer(
                best_model_info['model'], 
                best_model_info['feature_names']
            )
            
            # Initialize explainers with training data
            # Note: In a real scenario, you'd use actual training data here
            print(f"\nFeature importance for {best_model_name}:")
            print("(Note: SHAP explanations require training data)")
            
            # Display feature names
            print(f"\nTotal features: {len(best_model_info['feature_names'])}")
            print("Top 10 features:")
            for i, feature in enumerate(best_model_info['feature_names'][:10], 1):
                print(f"  {i:2d}. {feature}")
        
        # Save results summary
        summary_file = project_root / "data" / "models" / "training_summary.txt"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w') as f:
            f.write("CREDIT RISK MODEL TRAINING SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Training Date: {results['models'][list(results['models'].keys())[0]]['training_date']}\n")
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Best AUC: {best_auc:.4f}\n\n")
            f.write("Model Performance:\n")
            f.write(results['comparison'].to_string())
        
        logger.info(f"Training summary saved to {summary_file}")
        logger.info("Credit risk model training completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
