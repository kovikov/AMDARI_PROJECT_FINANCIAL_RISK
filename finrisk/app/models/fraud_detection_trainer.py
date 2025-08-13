"""
Fraud detection model training module for FinRisk application.
Implements Isolation Forest and One-Class SVM for anomaly detection.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

# FinRisk modules
from app.config import get_settings
from app.features.preprocessing import FinancialFeatureEngineer, FeatureTransformer
from app.infra.db import get_db_session, read_sql_query
from app.monitoring.mlflow_tracker import mlflow_tracker, log_fraud_detection_model

# Configure logging
logger = logging.getLogger(__name__)


class FraudDetectionMetrics:
    """Fraud detection specific metrics calculator."""
    
    @staticmethod
    def calculate_precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        """
        Calculate precision at top k predictions.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores
            k: Number of top predictions to consider
            
        Returns:
            Precision at k
        """
        # Get indices of top k predictions
        top_k_indices = np.argsort(y_scores)[-k:]
        top_k_labels = y_true[top_k_indices]
        
        return np.sum(top_k_labels) / k
    
    @staticmethod
    def calculate_lift_at_percentile(y_true: np.ndarray, y_scores: np.ndarray, 
                                   percentile: float = 0.05) -> float:
        """
        Calculate lift at given percentile.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores
            percentile: Percentile to calculate lift at
            
        Returns:
            Lift value
        """
        # Calculate threshold for percentile
        threshold = np.percentile(y_scores, (1 - percentile) * 100)
        
        # Calculate metrics
        baseline_fraud_rate = np.mean(y_true)
        fraud_rate_at_percentile = np.mean(y_true[y_scores >= threshold])
        
        lift = fraud_rate_at_percentile / baseline_fraud_rate if baseline_fraud_rate > 0 else 0
        return lift
    
    @staticmethod
    def calculate_detection_rate_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, 
                                      target_fpr: float = 0.01) -> float:
        """
        Calculate detection rate at specific false positive rate.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores
            target_fpr: Target false positive rate
            
        Returns:
            Detection rate
        """
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Find closest FPR to target
        idx = np.argmin(np.abs(fpr - target_fpr))
        detection_rate = tpr[idx]
        
        return detection_rate


class FraudDetectionModelTrainer:
    """Fraud detection model trainer with anomaly detection algorithms."""
    
    def __init__(self):
        self.settings = get_settings()
        self.models = {}
        self.feature_transformer = None
        self.feature_engineer = FinancialFeatureEngineer()
        self.mlflow_tracker = mlflow_tracker
        
        # Model configurations
        self.model_configs = {
            'isolation_forest': {
                'model': IsolationForest,
                'params': {
                    'random_state': 42,
                    'n_jobs': -1,
                    'contamination': 0.1  # Expected fraud rate
                },
                'search_params': {
                    'n_estimators': [100, 200, 300],
                    'max_samples': [0.5, 0.7, 1.0],
                    'max_features': [0.5, 0.7, 1.0],
                    'contamination': [0.05, 0.1, 0.15]
                }
            },
            'one_class_svm': {
                'model': OneClassSVM,
                'params': {
                    'kernel': 'rbf',
                    'gamma': 'scale'
                },
                'search_params': {
                    'nu': [0.05, 0.1, 0.15, 0.2],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'linear', 'poly']
                }
            },
            'elliptic_envelope': {
                'model': EllipticEnvelope,
                'params': {
                    'random_state': 42,
                    'contamination': 0.1
                },
                'search_params': {
                    'contamination': [0.05, 0.1, 0.15],
                    'support_fraction': [None, 0.8, 0.9, 0.95]
                }
            }
        }
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare fraud detection training data.
        
        Returns:
            Feature matrix and target variable
        """
        logger.info("Loading fraud detection training data...")
        
        # Load data from database
        transaction_query = "SELECT * FROM finrisk.transaction_data"
        customer_query = "SELECT * FROM finrisk.customer_profiles"
        
        transaction_df = read_sql_query(transaction_query)
        customer_df = read_sql_query(customer_query)
        
        logger.info(f"Loaded {len(transaction_df)} transactions from {len(customer_df)} customers")
        
        # Create fraud features
        features_df = self.feature_engineer.create_fraud_features(
            transaction_df, customer_df
        )
        
        # Prepare feature matrix and target
        exclude_cols = [
            'transaction_id', 'customer_id', 'fraud_flag', 'investigation_status',
            'transaction_date', 'created_at'
        ]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Handle missing values and infinite values
        features_df[feature_cols] = features_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        features_df[feature_cols] = features_df[feature_cols].fillna(features_df[feature_cols].median())
        
        X = features_df[feature_cols]
        y = features_df['fraud_flag']
        
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} transactions")
        logger.info(f"Fraud rate: {y.mean():.4%}")
        
        return X, y
    
    def train_anomaly_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train a specific anomaly detection model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features (legitimate transactions only)
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training {model_name} model...")
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        
        # For anomaly detection, train only on legitimate transactions
        X_train_legitimate = X_train[y_train == 0]
        logger.info(f"Training on {len(X_train_legitimate)} legitimate transactions")
        
        # Initialize model
        if hyperparameter_tuning:
            best_model, best_params = self._tune_hyperparameters(
                config, X_train_legitimate, X_val, y_val
            )
        else:
            best_model = config['model'](**config['params'])
            best_model.fit(X_train_legitimate)
            best_params = config['params']
        
        # Evaluate model
        metrics = self._evaluate_fraud_model(best_model, X_val, y_val, model_name)
        
        # Store model
        self.models[model_name] = {
            'model': best_model,
            'best_params': best_params,
            'metrics': metrics,
            'feature_names': X_train.columns.tolist()
        }
        
        logger.info(f"Completed training {model_name} - AUC: {metrics.get('auc', 'N/A')}")
        return self.models[model_name]
    
    def _tune_hyperparameters(self, config: Dict, X_train: pd.DataFrame,
                            X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, Dict]:
        """
        Tune hyperparameters for anomaly detection model.
        
        Args:
            config: Model configuration
            X_train: Training features (legitimate only)
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Best model and parameters
        """
        from itertools import product
        
        best_score = -np.inf
        best_model = None
        best_params = None
        
        # Generate parameter combinations
        param_names = list(config['search_params'].keys())
        param_values = list(config['search_params'].values())
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            params.update(config['params'])  # Add base parameters
            
            try:
                # Train model
                model = config['model'](**params)
                model.fit(X_train)
                
                # Evaluate on validation set
                if hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_val)
                elif hasattr(model, 'score_samples'):
                    y_scores = model.score_samples(X_val)
                else:
                    continue
                
                # Calculate AUC (higher is better)
                auc = roc_auc_score(y_val, y_scores)
                
                if auc > best_score:
                    best_score = auc
                    best_model = model
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Failed parameter combination {params}: {e}")
                continue
        
        logger.info(f"Best parameters: {best_params}, Best AUC: {best_score:.4f}")
        return best_model, best_params
    
    def _evaluate_fraud_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                            model_name: str) -> Dict[str, float]:
        """
        Evaluate fraud detection model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Model name for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get anomaly scores
        if hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_test)
        elif hasattr(model, 'score_samples'):
            y_scores = model.score_samples(X_test)
        else:
            raise ValueError("Model must have decision_function or score_samples method")
        
        # Predictions (anomalies are labeled as -1, convert to 1)
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred == -1).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        try:
            metrics['auc'] = roc_auc_score(y_test, y_scores)
            metrics['average_precision'] = average_precision_score(y_test, y_scores)
        except:
            metrics['auc'] = 0.5
            metrics['average_precision'] = y_test.mean()
        
        metrics['precision'] = precision_score(y_test, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred_binary, zero_division=0)
        metrics['f1_score'] = f1_score(y_test, y_pred_binary, zero_division=0)
        
        # Fraud-specific metrics
        metrics['precision_at_1pct'] = FraudDetectionMetrics.calculate_precision_at_k(
            y_test.values, y_scores, max(1, int(0.01 * len(y_test)))
        )
        metrics['lift_at_5pct'] = FraudDetectionMetrics.calculate_lift_at_percentile(
            y_test.values, y_scores, 0.05
        )
        metrics['detection_rate_at_1pct_fpr'] = FraudDetectionMetrics.calculate_detection_rate_at_fpr(
            y_test.values, y_scores, 0.01
        )
        
        # Log experiment to MLflow
        try:
            run_id = log_fraud_detection_model(
                model=model,
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                hyperparameters=model.get_params(),
                metrics=metrics,
                anomaly_scores=y_scores,
                model_type="sklearn"
            )
            logger.info(f"Fraud detection experiment logged to MLflow: {run_id}")
        except Exception as e:
            logger.warning(f"Failed to log experiment to MLflow: {e}")
        
        return metrics
    
    def train_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Train ensemble fraud detection model combining multiple algorithms.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with ensemble model and metrics
        """
        logger.info("Training ensemble fraud detection model...")
        
        # Train individual models
        individual_models = {}
        for model_name in self.model_configs.keys():
            try:
                model_result = self.train_anomaly_model(
                    model_name, X_train, y_train, X_val, y_val, hyperparameter_tuning=False
                )
                individual_models[model_name] = model_result['model']
            except Exception as e:
                logger.error(f"Failed to train {model_name} for ensemble: {e}")
        
        # Create ensemble predictor
        class FraudEnsemble:
            def __init__(self, models: Dict[str, Any]):
                self.models = models
                self.weights = {name: 1.0 for name in models.keys()}  # Equal weights
            
            def predict(self, X):
                predictions = []
                for name, model in self.models.items():
                    pred = model.predict(X)
                    # Convert -1 to 1 for anomalies
                    pred_binary = (pred == -1).astype(int)
                    predictions.append(pred_binary * self.weights[name])
                
                # Majority vote with weights
                ensemble_pred = np.sum(predictions, axis=0)
                threshold = sum(self.weights.values()) / 2
                return (ensemble_pred >= threshold).astype(int)
            
            def decision_function(self, X):
                scores = []
                for name, model in self.models.items():
                    if hasattr(model, 'decision_function'):
                        score = model.decision_function(X)
                    elif hasattr(model, 'score_samples'):
                        score = model.score_samples(X)
                    else:
                        continue
                    scores.append(score * self.weights[name])
                
                return np.mean(scores, axis=0) if scores else np.zeros(X.shape[0])
        
        # Create ensemble
        ensemble_model = FraudEnsemble(individual_models)
        
        # Evaluate ensemble
        ensemble_metrics = self._evaluate_fraud_model(
            ensemble_model, X_val, y_val, "ensemble"
        )
        
        self.models['ensemble'] = {
            'model': ensemble_model,
            'individual_models': individual_models,
            'metrics': ensemble_metrics,
            'feature_names': X_train.columns.tolist()
        }
        
        logger.info(f"Ensemble model AUC: {ensemble_metrics.get('auc', 'N/A'):.4f}")
        return self.models['ensemble']
    
    def train_all_models(self, test_size: float = 0.2,
                        hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train all fraud detection models and compare performance.
        
        Args:
            test_size: Proportion of data for testing
            hyperparameter_tuning: Whether to tune hyperparameters
            
        Returns:
            Dictionary with all trained models and comparison
        """
        logger.info("Starting fraud detection model training pipeline...")
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Initialize feature transformer
        self.feature_transformer = FeatureTransformer(feature_type='fraud')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Split training into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        # Fit feature transformer
        X_train_transformed = self.feature_transformer.fit_transform(X_train)
        X_val_transformed = self.feature_transformer.transform(X_val)
        X_test_transformed = self.feature_transformer.transform(X_test)
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train individual models
        results = {}
        for model_name in self.model_configs.keys():
            try:
                model_result = self.train_anomaly_model(
                    model_name, X_train_transformed, y_train,
                    X_val_transformed, y_val, hyperparameter_tuning
                )
                results[model_name] = model_result
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Train ensemble model
        try:
            ensemble_result = self.train_ensemble_model(
                X_train_transformed, y_train, X_val_transformed, y_val
            )
            results['ensemble'] = ensemble_result
        except Exception as e:
            logger.error(f"Failed to train ensemble model: {e}")
        
        # Final evaluation on test set
        test_results = {}
        for model_name, model_info in results.items():
            try:
                test_metrics = self._evaluate_fraud_model(
                    model_info['model'], X_test_transformed, y_test, f"{model_name}_test"
                )
                test_results[model_name] = test_metrics
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name} on test set: {e}")
        
        # Model comparison
        if test_results:
            comparison_df = pd.DataFrame(test_results).T
            comparison_df = comparison_df.round(4)
            
            logger.info("Fraud Model Performance Comparison:")
            logger.info("\n" + comparison_df.to_string())
        else:
            comparison_df = pd.DataFrame()
        
        # Save models and transformer
        self._save_models(results)
        
        return {
            'models': results,
            'test_results': test_results,
            'comparison': comparison_df,
            'feature_transformer': self.feature_transformer,
            'feature_names': X_train.columns.tolist()
        }
    
    def _save_models(self, models: Dict[str, Any]) -> None:
        """Save trained fraud detection models to disk."""
        model_path = Path(self.settings.paths.model_store_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model_info in models.items():
            # Save model
            model_file = model_path / f"fraud_{model_name}_{timestamp}.joblib"
            joblib.dump(model_info['model'], model_file)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'training_date': datetime.now().isoformat(),
                'metrics': model_info['metrics'],
                'feature_names': model_info['feature_names']
            }
            
            if 'best_params' in model_info:
                metadata['best_params'] = model_info['best_params']
            
            metadata_file = model_path / f"fraud_{model_name}_{timestamp}_metadata.json"
            with open(metadata_file, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
        
        # Save feature transformer
        transformer_file = model_path / f"fraud_feature_transformer_{timestamp}.joblib"
        joblib.dump(self.feature_transformer, transformer_file)
        
        logger.info(f"Saved fraud detection models to {model_path}")


class FraudRuleEngine:
    """Rule-based fraud detection engine to complement ML models."""
    
    def __init__(self):
        self.rules = {}
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize fraud detection rules."""
        self.rules = {
            'high_amount_rule': {
                'condition': lambda row: row['amount'] > 10000,
                'severity': 'HIGH',
                'message': 'Transaction amount exceeds £10,000'
            },
            'velocity_rule': {
                'condition': lambda row: row.get('rolling_24h_count', 0) > 20,
                'severity': 'MEDIUM',
                'message': 'High transaction velocity (>20 in 24h)'
            },
            'unusual_time_rule': {
                'condition': lambda row: row.get('is_night', 0) == 1 and row['amount'] > 1000,
                'severity': 'MEDIUM',
                'message': 'Large transaction during unusual hours'
            },
            'new_location_rule': {
                'condition': lambda row: row.get('new_location', 0) == 1 and row['amount'] > 5000,
                'severity': 'HIGH',
                'message': 'Large transaction from new location'
            },
            'round_amount_rule': {
                'condition': lambda row: row.get('is_round_amount', 0) == 1 and row['amount'] >= 1000,
                'severity': 'LOW',
                'message': 'Round amount transaction'
            }
        }
    
    def evaluate_transaction(self, transaction_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate transaction against fraud rules.
        
        Args:
            transaction_features: Dictionary with transaction features
            
        Returns:
            Dictionary with rule evaluation results
        """
        triggered_rules = []
        max_severity = 'LOW'
        
        for rule_name, rule in self.rules.items():
            try:
                if rule['condition'](transaction_features):
                    triggered_rules.append({
                        'rule_name': rule_name,
                        'severity': rule['severity'],
                        'message': rule['message']
                    })
                    
                    # Update max severity
                    severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                    if severities.index(rule['severity']) > severities.index(max_severity):
                        max_severity = rule['severity']
                        
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule_name}: {e}")
        
        return {
            'triggered_rules': triggered_rules,
            'rule_count': len(triggered_rules),
            'max_severity': max_severity,
            'requires_review': len(triggered_rules) > 0
        }


class FraudModelExplainer:
    """Model interpretability for fraud detection models."""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
    
    def explain_anomaly_score(self, X_instance: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain anomaly score for a transaction.
        
        Args:
            X_instance: Single transaction features
            
        Returns:
            Dictionary with explanations
        """
        explanations = {}
        
        # Get anomaly score
        if hasattr(self.model, 'decision_function'):
            anomaly_score = self.model.decision_function(X_instance)[0]
        elif hasattr(self.model, 'score_samples'):
            anomaly_score = self.model.score_samples(X_instance)[0]
        else:
            anomaly_score = 0.0
        
        explanations['anomaly_score'] = float(anomaly_score)
        explanations['is_anomaly'] = anomaly_score < 0  # Negative scores indicate anomalies
        
        # Feature contributions (simplified approach)
        feature_values = X_instance.iloc[0].to_dict()
        
        # Calculate feature deviations from training mean (requires training stats)
        # This is a simplified approach - in practice, you'd use SHAP or LIME
        feature_contributions = {}
        for feature, value in feature_values.items():
            # Normalize contribution based on absolute value
            contribution = abs(value) if not np.isnan(value) else 0
            feature_contributions[feature] = contribution
        
        # Sort by contribution
        sorted_contributions = sorted(
            feature_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        explanations['feature_contributions'] = dict(sorted_contributions[:10])
        explanations['text_explanation'] = self._generate_fraud_explanation(explanations)
        
        return explanations
    
    def _generate_fraud_explanation(self, explanations: Dict[str, Any]) -> str:
        """Generate human-readable fraud explanation."""
        text_parts = []
        
        # Risk level
        anomaly_score = explanations['anomaly_score']
        is_anomaly = explanations['is_anomaly']
        
        if is_anomaly:
            if anomaly_score < -0.5:
                text_parts.append("HIGH FRAUD RISK - Strong anomaly detected")
            else:
                text_parts.append("MEDIUM FRAUD RISK - Moderate anomaly detected")
        else:
            text_parts.append("LOW FRAUD RISK - Transaction appears normal")
        
        text_parts.append(f"Anomaly score: {anomaly_score:.3f}")
        
        # Top contributing features
        if 'feature_contributions' in explanations:
            top_features = list(explanations['feature_contributions'].items())[:3]
            if top_features:
                text_parts.append("\nKey factors:")
                for feature, contribution in top_features:
                    text_parts.append(f"- {feature}: {contribution:.3f}")
        
        return "\n".join(text_parts)


def train_fraud_models(hyperparameter_tuning: bool = True) -> Dict[str, Any]:
    """
    Main function to train all fraud detection models.
    
    Args:
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        Dictionary with training results
    """
    trainer = FraudDetectionModelTrainer()
    results = trainer.train_all_models(hyperparameter_tuning=hyperparameter_tuning)
    
    logger.info("Fraud detection model training completed successfully")
    return results


def create_fraud_prediction_pipeline(model_name: str = 'ensemble') -> Tuple[Any, Any]:
    """
    Create fraud prediction pipeline with trained model.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        Tuple of (model, feature_transformer)
    """
    # Load latest trained model
    model_path = Path(get_settings().paths.model_store_path)
    
    # Find latest model files
    model_files = list(model_path.glob(f"fraud_{model_name}_*.joblib"))
    transformer_files = list(model_path.glob("fraud_feature_transformer_*.joblib"))
    
    if not model_files or not transformer_files:
        raise FileNotFoundError(f"No trained {model_name} model found")
    
    # Load latest files
    latest_model_file = max(model_files, key=lambda f: f.stat().st_mtime)
    latest_transformer_file = max(transformer_files, key=lambda f: f.stat().st_mtime)
    
    model = joblib.load(latest_model_file)
    transformer = joblib.load(latest_transformer_file)
    
    logger.info(f"Loaded fraud model: {latest_model_file.name}")
    
    return model, transformer


if __name__ == "__main__":
    # Run fraud model training
    results = train_fraud_models(hyperparameter_tuning=True)
    print("Fraud detection training completed!")
    if 'comparison' in results and not results['comparison'].empty:
        print(results['comparison'])
