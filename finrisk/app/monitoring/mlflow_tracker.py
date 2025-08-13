#!/usr/bin/env python3
"""
MLflow tracking module for FinRisk application.
Handles experiment tracking, model logging, and metrics recording.
"""

import os
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import joblib
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinRiskMLflowTracker:
    """MLflow tracker for FinRisk experiments and model management."""
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = "finrisk-experiments"):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        self.experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "finrisk-experiments")
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Get or create experiment
        self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = self.experiment.experiment_id
        
        logger.info(f"MLflow tracker initialized with experiment: {self.experiment_name}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name of the run
            tags: Additional tags for the run
            
        Returns:
            Active MLflow run
        """
        tags = tags or {}
        tags.update({
            "project": "finrisk",
            "timestamp": datetime.now().isoformat()
        })
        
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
    
    def log_credit_risk_experiment(
        self,
        model,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        feature_importance: Dict[str, float] = None,
        model_type: str = "sklearn"
    ):
        """
        Log credit risk model experiment.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            hyperparameters: Model hyperparameters
            metrics: Evaluation metrics
            feature_importance: Feature importance scores
            model_type: Type of model (sklearn, xgboost, lightgbm)
        """
        run_name = f"credit_risk_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self.start_run(run_name=run_name, tags={"model_type": "credit_risk", "algorithm": model_name}) as run:
            # Log parameters
            mlflow.log_params(hyperparameters)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log feature importance
            if feature_importance:
                mlflow.log_dict(feature_importance, "feature_importance.json")
            
            # Log model
            if model_type == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Log training data info
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            mlflow.log_metric("n_features", X_train.shape[1])
            
            # Log feature names
            mlflow.log_dict({"feature_names": list(X_train.columns)}, "feature_info.json")
            
            # Log data distributions
            self._log_data_distributions(X_train, X_test, y_train, y_test)
            
            logger.info(f"Credit risk experiment logged: {run.info.run_id}")
            return run.info.run_id
    
    def log_fraud_detection_experiment(
        self,
        model,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        anomaly_scores: np.ndarray = None,
        model_type: str = "sklearn"
    ):
        """
        Log fraud detection model experiment.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            hyperparameters: Model hyperparameters
            metrics: Evaluation metrics
            anomaly_scores: Anomaly scores from the model
            model_type: Type of model (sklearn, xgboost, lightgbm)
        """
        run_name = f"fraud_detection_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self.start_run(run_name=run_name, tags={"model_type": "fraud_detection", "algorithm": model_name}) as run:
            # Log parameters
            mlflow.log_params(hyperparameters)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            if model_type == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Log training data info
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            mlflow.log_metric("n_features", X_train.shape[1])
            mlflow.log_metric("fraud_rate_train", y_train.mean())
            mlflow.log_metric("fraud_rate_test", y_test.mean())
            
            # Log feature names
            mlflow.log_dict({"feature_names": list(X_train.columns)}, "feature_info.json")
            
            # Log anomaly scores distribution
            if anomaly_scores is not None:
                anomaly_stats = {
                    "mean": float(np.mean(anomaly_scores)),
                    "std": float(np.std(anomaly_scores)),
                    "min": float(np.min(anomaly_scores)),
                    "max": float(np.max(anomaly_scores)),
                    "percentiles": {
                        "25": float(np.percentile(anomaly_scores, 25)),
                        "50": float(np.percentile(anomaly_scores, 50)),
                        "75": float(np.percentile(anomaly_scores, 75)),
                        "95": float(np.percentile(anomaly_scores, 95)),
                        "99": float(np.percentile(anomaly_scores, 99))
                    }
                }
                mlflow.log_dict(anomaly_stats, "anomaly_scores_stats.json")
            
            # Log data distributions
            self._log_data_distributions(X_train, X_test, y_train, y_test)
            
            logger.info(f"Fraud detection experiment logged: {run.info.run_id}")
            return run.info.run_id
    
    def log_model_comparison(
        self,
        models: Dict[str, Any],
        model_names: List[str],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metrics: Dict[str, Dict[str, float]],
        comparison_name: str = "model_comparison"
    ):
        """
        Log model comparison experiment.
        
        Args:
            models: Dictionary of trained models
            model_names: List of model names
            X_test: Test features
            y_test: Test labels
            metrics: Dictionary of metrics for each model
            comparison_name: Name of the comparison experiment
        """
        run_name = f"{comparison_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self.start_run(run_name=run_name, tags={"experiment_type": "model_comparison"}) as run:
            # Log comparison metrics
            for model_name in model_names:
                if model_name in metrics:
                    for metric_name, metric_value in metrics[model_name].items():
                        mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
            
            # Log test data info
            mlflow.log_metric("test_samples", len(X_test))
            mlflow.log_metric("n_features", X_test.shape[1])
            
            # Log feature names
            mlflow.log_dict({"feature_names": list(X_test.columns)}, "feature_info.json")
            
            # Log comparison summary
            comparison_summary = {
                "models_compared": model_names,
                "best_model_by_metric": {}
            }
            
            # Find best model for each metric
            for metric_name in set().union(*[set(metrics[model].keys()) for model in model_names if model in metrics]):
                best_model = max(
                    model_names,
                    key=lambda m: metrics.get(m, {}).get(metric_name, float('-inf'))
                )
                comparison_summary["best_model_by_metric"][metric_name] = best_model
            
            mlflow.log_dict(comparison_summary, "comparison_summary.json")
            
            logger.info(f"Model comparison logged: {run.info.run_id}")
            return run.info.run_id
    
    def log_hyperparameter_tuning(
        self,
        param_grid: Dict[str, List],
        cv_results: Dict[str, List],
        best_params: Dict[str, Any],
        best_score: float,
        tuning_name: str = "hyperparameter_tuning"
    ):
        """
        Log hyperparameter tuning results.
        
        Args:
            param_grid: Parameter grid used for tuning
            cv_results: Cross-validation results
            best_params: Best parameters found
            best_score: Best score achieved
            tuning_name: Name of the tuning experiment
        """
        run_name = f"{tuning_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self.start_run(run_name=run_name, tags={"experiment_type": "hyperparameter_tuning"}) as run:
            # Log parameter grid
            mlflow.log_dict(param_grid, "parameter_grid.json")
            
            # Log best parameters
            mlflow.log_params(best_params)
            mlflow.log_metric("best_score", best_score)
            
            # Log CV results summary
            cv_summary = {
                "mean_test_score": float(np.mean(cv_results.get("mean_test_score", []))),
                "std_test_score": float(np.std(cv_results.get("mean_test_score", []))),
                "n_splits": len(cv_results.get("split0_test_score", [])),
                "n_combinations": len(cv_results.get("mean_test_score", []))
            }
            mlflow.log_dict(cv_summary, "cv_summary.json")
            
            # Log full CV results
            mlflow.log_dict(cv_results, "cv_results.json")
            
            logger.info(f"Hyperparameter tuning logged: {run.info.run_id}")
            return run.info.run_id
    
    def log_data_quality_report(
        self,
        data_quality_metrics: Dict[str, Any],
        dataset_name: str = "dataset"
    ):
        """
        Log data quality report.
        
        Args:
            data_quality_metrics: Data quality metrics
            dataset_name: Name of the dataset
        """
        run_name = f"data_quality_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self.start_run(run_name=run_name, tags={"experiment_type": "data_quality"}) as run:
            # Log data quality metrics
            for metric_name, metric_value in data_quality_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
                else:
                    mlflow.log_dict({metric_name: metric_value}, f"{metric_name}.json")
            
            logger.info(f"Data quality report logged: {run.info.run_id}")
            return run.info.run_id
    
    def load_model(self, run_id: str, model_path: str = "model") -> Any:
        """
        Load a logged model.
        
        Args:
            run_id: MLflow run ID
            model_path: Path to the model within the run
            
        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/{model_path}"
        return mlflow.sklearn.load_model(model_uri)
    
    def get_best_run(self, metric: str, greater_is_better: bool = True, model_type: str = None) -> Optional[str]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric name to optimize
            greater_is_better: Whether higher values are better
            model_type: Filter by model type (credit_risk, fraud_detection)
            
        Returns:
            Best run ID or None
        """
        experiment = mlflow.get_experiment(self.experiment_id)
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.model_type = '{model_type}'" if model_type else None,
            order_by=[f"metrics.{metric} DESC" if greater_is_better else f"metrics.{metric} ASC"]
        )
        
        if runs.empty:
            return None
        
        return runs.iloc[0]["run_id"]
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary of all experiments.
        
        Returns:
            Experiment summary
        """
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        
        summary = {
            "total_runs": len(runs),
            "experiment_name": self.experiment_name,
            "model_types": runs["tags.model_type"].value_counts().to_dict(),
            "algorithms": runs["tags.algorithm"].value_counts().to_dict(),
            "latest_runs": []
        }
        
        # Get latest runs
        for _, run in runs.head(10).iterrows():
            summary["latest_runs"].append({
                "run_id": run["run_id"],
                "run_name": run["tags.mlflow.runName"],
                "model_type": run["tags.model_type"],
                "algorithm": run["tags.algorithm"],
                "status": run["status"],
                "start_time": run["start_time"]
            })
        
        return summary
    
    def _log_data_distributions(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ):
        """Log data distribution statistics."""
        # Log target distribution
        train_target_dist = y_train.value_counts().to_dict()
        test_target_dist = y_test.value_counts().to_dict()
        
        mlflow.log_dict({
            "train_target_distribution": train_target_dist,
            "test_target_distribution": test_target_dist
        }, "target_distributions.json")
        
        # Log feature statistics
        feature_stats = {}
        for col in X_train.columns:
            feature_stats[col] = {
                "train_mean": float(X_train[col].mean()),
                "train_std": float(X_train[col].std()),
                "test_mean": float(X_test[col].mean()),
                "test_std": float(X_test[col].std()),
                "train_missing": float(X_train[col].isnull().sum()),
                "test_missing": float(X_test[col].isnull().sum())
            }
        
        mlflow.log_dict(feature_stats, "feature_statistics.json")


class FinRiskModelRegistry:
    """Model registry for FinRisk models."""
    
    def __init__(self, registry_uri: str = None):
        """
        Initialize model registry.
        
        Args:
            registry_uri: Model registry URI
        """
        self.registry_uri = registry_uri or os.getenv("MLFLOW_REGISTRY_URI", "sqlite:///mlflow.db")
        mlflow.set_registry_uri(self.registry_uri)
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        model_path: str = "model",
        description: str = None
    ) -> str:
        """
        Register a model from a run.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            model_path: Path to the model within the run
            description: Model description
            
        Returns:
            Registered model version
        """
        model_uri = f"runs:/{run_id}/{model_path}"
        
        try:
            # Create model if it doesn't exist
            try:
                mlflow.register_model(model_uri, model_name)
            except Exception:
                # Model might already exist, try to create new version
                pass
            
            # Get the latest version
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
            
            # Update description if provided
            if description:
                client.update_registered_model(
                    name=model_name,
                    description=description
                )
            
            logger.info(f"Model registered: {model_name} v{latest_version.version}")
            return latest_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str = "Production"
    ):
        """
        Promote a model to a specific stage.
        
        Args:
            model_name: Name of the registered model
            version: Model version
            stage: Target stage (Staging, Production, Archived)
        """
        client = mlflow.tracking.MlflowClient()
        
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Model {model_name} v{version} promoted to {stage}")
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    def load_production_model(self, model_name: str) -> Any:
        """
        Load the production version of a model.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            Production model
        """
        model_uri = f"models:/{model_name}/Production"
        return mlflow.sklearn.load_model(model_uri)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Returns:
            List of model information
        """
        client = mlflow.tracking.MlflowClient()
        models = client.list_registered_models()
        
        model_info = []
        for model in models:
            versions = client.get_latest_versions(model.name)
            model_info.append({
                "name": model.name,
                "description": model.description,
                "latest_version": versions[0].version if versions else None,
                "latest_stage": versions[0].current_stage if versions else None,
                "creation_timestamp": model.creation_timestamp
            })
        
        return model_info


# Global tracker instance
mlflow_tracker = FinRiskMLflowTracker()
model_registry = FinRiskModelRegistry()


def log_credit_risk_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, float],
    feature_importance: Dict[str, float] = None,
    model_type: str = "sklearn"
) -> str:
    """Convenience function to log credit risk model."""
    return mlflow_tracker.log_credit_risk_experiment(
        model, model_name, X_train, X_test, y_train, y_test,
        hyperparameters, metrics, feature_importance, model_type
    )


def log_fraud_detection_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, float],
    anomaly_scores: np.ndarray = None,
    model_type: str = "sklearn"
) -> str:
    """Convenience function to log fraud detection model."""
    return mlflow_tracker.log_fraud_detection_experiment(
        model, model_name, X_train, X_test, y_train, y_test,
        hyperparameters, metrics, anomaly_scores, model_type
    )


def register_model_to_production(run_id: str, model_name: str, description: str = None) -> str:
    """Convenience function to register and promote model to production."""
    version = model_registry.register_model(run_id, model_name, description=description)
    model_registry.promote_model(model_name, version, "Production")
    return version
