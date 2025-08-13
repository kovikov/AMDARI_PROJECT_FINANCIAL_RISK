"""
MLflow utilities for FinRisk application.
Handles experiment tracking, model logging, and metric monitoring.
"""

import os
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np

from app.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracking for FinRisk models."""
    
    def __init__(self):
        self.settings = get_settings()
        self._setup_mlflow()
        self.current_experiment = None
    
    def _setup_mlflow(self):
        """Setup MLflow configuration."""
        try:
            mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
            mlflow.set_registry_uri(self.settings.mlflow.backend_store_uri)
            
            # Set default artifact location
            if not os.path.exists(self.settings.mlflow.default_artifact_root):
                os.makedirs(self.settings.mlflow.default_artifact_root, exist_ok=True)
            
            logger.info(f"MLflow tracking URI: {self.settings.mlflow.tracking_uri}")
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
    
    def create_experiment(self, experiment_name: str, 
                         description: str = "") -> str:
        """
        Create or get MLflow experiment.
        
        Args:
            experiment_name: Name of the experiment
            description: Experiment description
            
        Returns:
            Experiment ID
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=self.settings.mlflow.default_artifact_root,
                    tags={"description": description}
                )
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
            
            self.current_experiment = experiment_name
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment {experiment_name}: {e}")
            return None
    
    def start_run(self, run_name: str, experiment_name: str = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name of the run
            experiment_name: Name of the experiment (optional)
            
        Returns:
            Active MLflow run
        """
        try:
            if experiment_name:
                self.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
            
            run = mlflow.start_run(run_name=run_name)
            logger.info(f"Started MLflow run: {run_name}")
            return run
            
        except Exception as e:
            logger.error(f"Failed to start run {run_name}: {e}")
            return None
    
    def log_model_metrics(self, model_name: str, metrics: Dict[str, float],
                         model_type: str = "classification") -> None:
        """
        Log model metrics to MLflow.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics to log
            model_type: Type of model (classification, regression, etc.)
        """
        try:
            # Log basic metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", value)
            
            # Log model type as parameter
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("model_name", model_name)
            
            # Log timestamp
            mlflow.log_param("training_timestamp", datetime.now().isoformat())
            
            logger.info(f"Logged {len(metrics)} metrics for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics for {model_name}: {e}")
    
    def log_model_parameters(self, model_name: str, parameters: Dict[str, Any]) -> None:
        """
        Log model parameters to MLflow.
        
        Args:
            model_name: Name of the model
            parameters: Dictionary of parameters to log
        """
        try:
            for param_name, value in parameters.items():
                mlflow.log_param(f"{model_name}_{param_name}", value)
            
            logger.info(f"Logged {len(parameters)} parameters for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log parameters for {model_name}: {e}")
    
    def log_model(self, model, model_name: str, model_type: str = "sklearn") -> None:
        """
        Log trained model to MLflow.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            model_type: Type of model (sklearn, xgboost, lightgbm)
        """
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, f"{model_name}_model")
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(model, f"{model_name}_model")
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(model, f"{model_name}_model")
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return
            
            logger.info(f"Logged {model_type} model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log model {model_name}: {e}")
    
    def log_feature_importance(self, feature_names: List[str], 
                             importance_scores: List[float],
                             model_name: str) -> None:
        """
        Log feature importance to MLflow.
        
        Args:
            feature_names: List of feature names
            importance_scores: List of importance scores
            model_name: Name of the model
        """
        try:
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            # Save as CSV artifact
            importance_file = f"{model_name}_feature_importance.csv"
            importance_df.to_csv(importance_file, index=False)
            mlflow.log_artifact(importance_file)
            
            # Clean up temporary file
            os.remove(importance_file)
            
            logger.info(f"Logged feature importance for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log feature importance for {model_name}: {e}")
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str) -> None:
        """
        Log confusion matrix to MLflow.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save and log
            cm_file = f"{model_name}_confusion_matrix.png"
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(cm_file)
            
            # Clean up
            plt.close()
            os.remove(cm_file)
            
            logger.info(f"Logged confusion matrix for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log confusion matrix for {model_name}: {e}")
    
    def log_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                     model_name: str) -> None:
        """
        Log ROC curve to MLflow.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
        """
        try:
            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Create plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            
            # Save and log
            roc_file = f"{model_name}_roc_curve.png"
            plt.savefig(roc_file, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(roc_file)
            
            # Clean up
            plt.close()
            os.remove(roc_file)
            
            logger.info(f"Logged ROC curve for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log ROC curve for {model_name}: {e}")
    
    def log_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                 model_name: str) -> None:
        """
        Log precision-recall curve to MLflow.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
        """
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            import matplotlib.pyplot as plt
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            
            # Create plot
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AP = {avg_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend()
            plt.grid(True)
            
            # Save and log
            pr_file = f"{model_name}_precision_recall_curve.png"
            plt.savefig(pr_file, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(pr_file)
            
            # Clean up
            plt.close()
            os.remove(pr_file)
            
            logger.info(f"Logged precision-recall curve for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log precision-recall curve for {model_name}: {e}")
    
    def log_model_comparison(self, comparison_df: pd.DataFrame, 
                           experiment_name: str) -> None:
        """
        Log model comparison results to MLflow.
        
        Args:
            comparison_df: DataFrame with model comparison results
            experiment_name: Name of the experiment
        """
        try:
            # Save comparison as CSV
            comparison_file = f"{experiment_name}_model_comparison.csv"
            comparison_df.to_csv(comparison_file)
            mlflow.log_artifact(comparison_file)
            
            # Log best model metrics
            best_model = comparison_df.loc[comparison_df['auc'].idxmax()]
            mlflow.log_metric("best_model_auc", best_model['auc'])
            mlflow.log_param("best_model_name", best_model.name)
            
            # Clean up
            os.remove(comparison_file)
            
            logger.info(f"Logged model comparison for {experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to log model comparison: {e}")
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Failed to end run: {e}")
    
    def get_experiment_runs(self, experiment_name: str) -> List[Dict[str, Any]]:
        """
        Get all runs for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            List of run information dictionaries
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                return []
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"]
            )
            
            return runs.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to get runs for {experiment_name}: {e}")
            return []
    
    def load_model(self, run_id: str, model_name: str, model_type: str = "sklearn"):
        """
        Load a logged model from MLflow.
        
        Args:
            run_id: MLflow run ID
            model_name: Name of the model
            model_type: Type of model (sklearn, xgboost, lightgbm)
            
        Returns:
            Loaded model object
        """
        try:
            model_uri = f"runs:/{run_id}/{model_name}_model"
            
            if model_type == "sklearn":
                return mlflow.sklearn.load_model(model_uri)
            elif model_type == "xgboost":
                return mlflow.xgboost.load_model(model_uri)
            elif model_type == "lightgbm":
                return mlflow.lightgbm.load_model(model_uri)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name} from run {run_id}: {e}")
            return None


def log_credit_risk_experiment(model_name: str, model, metrics: Dict[str, float],
                              parameters: Dict[str, Any], feature_names: List[str],
                              importance_scores: List[float] = None,
                              y_true: np.ndarray = None, y_pred: np.ndarray = None,
                              y_prob: np.ndarray = None) -> str:
    """
    Convenience function to log a complete credit risk experiment.
    
    Args:
        model_name: Name of the model
        model: Trained model object
        metrics: Model performance metrics
        parameters: Model parameters
        feature_names: List of feature names
        importance_scores: Feature importance scores (optional)
        y_true: True labels (optional)
        y_pred: Predicted labels (optional)
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Run ID
    """
    tracker = MLflowTracker()
    
    try:
        # Create experiment
        experiment_name = "credit_risk_modeling"
        tracker.create_experiment(experiment_name, "Credit risk model training experiments")
        
        # Start run
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = tracker.start_run(run_name, experiment_name)
        
        if run:
            # Log model and metrics
            tracker.log_model(model, model_name, "sklearn")
            tracker.log_model_metrics(model_name, metrics)
            tracker.log_model_parameters(model_name, parameters)
            
            # Log feature importance if available
            if importance_scores:
                tracker.log_feature_importance(feature_names, importance_scores, model_name)
            
            # Log evaluation plots if data available
            if y_true is not None and y_pred is not None:
                tracker.log_confusion_matrix(y_true, y_pred, model_name)
            
            if y_true is not None and y_prob is not None:
                tracker.log_roc_curve(y_true, y_prob, model_name)
                tracker.log_precision_recall_curve(y_true, y_prob, model_name)
            
            # End run
            tracker.end_run()
            
            return run.info.run_id
            
    except Exception as e:
        logger.error(f"Failed to log experiment for {model_name}: {e}")
        return None
