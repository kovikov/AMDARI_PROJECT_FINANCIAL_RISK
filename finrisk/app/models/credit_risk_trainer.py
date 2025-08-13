"""
Credit risk model training module for FinRisk application.
Implements XGBoost, Random Forest, and Logistic Regression models with comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

# Model interpretability
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

# FinRisk modules
from app.config import get_settings
from app.features.preprocessing import FinancialFeatureEngineer, FeatureTransformer
from app.infra.db import get_db_session, read_sql_query
from app.monitoring.mlflow_tracker import mlflow_tracker, log_credit_risk_model

# Configure logging
logger = logging.getLogger(__name__)


class CreditRiskMetrics:
    """Credit risk specific metrics calculator."""
    
    @staticmethod
    def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculate Kolmogorov-Smirnov statistic.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            
        Returns:
            KS statistic value
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ks_stat = np.max(tpr - fpr)
        return ks_stat
    
    @staticmethod
    def calculate_gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculate Gini coefficient.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            
        Returns:
            Gini coefficient
        """
        auc = roc_auc_score(y_true, y_prob)
        gini = 2 * auc - 1
        return gini
    
    @staticmethod
    def calculate_population_stability_index(expected: np.ndarray, 
                                           actual: np.ndarray, 
                                           bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            expected: Expected distribution (training)
            actual: Actual distribution (validation/test)
            bins: Number of bins for calculation
            
        Returns:
            PSI value
        """
        # Create bins
        breakpoints = np.linspace(0, 1, bins + 1)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Calculate distributions
        expected_counts = np.histogram(expected, breakpoints)[0]
        actual_counts = np.histogram(actual, breakpoints)[0]
        
        # Convert to percentages
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)
        
        # Avoid division by zero
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi


class CreditRiskModelTrainer:
    """Credit risk model trainer with multiple algorithms."""
    
    def __init__(self):
        self.settings = get_settings()
        self.models = {}
        self.feature_transformer = None
        self.feature_engineer = FinancialFeatureEngineer()
        self.mlflow_tracker = mlflow_tracker
        
        # Model configurations
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000,
                    'class_weight': 'balanced'
                },
                'search_params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'random_state': 42,
                    'n_jobs': -1,
                    'class_weight': 'balanced'
                },
                'search_params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'random_state': 42,
                    'eval_metric': 'logloss',
                    'use_label_encoder': False
                },
                'search_params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        }
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare credit risk training data.
        
        Returns:
            Feature matrix and target variable
        """
        logger.info("Loading credit risk training data...")
        
        # Load data from database
        customer_query = "SELECT * FROM finrisk.customer_profiles"
        bureau_query = "SELECT * FROM finrisk.credit_bureau_data"
        applications_query = """
            SELECT * FROM finrisk.credit_applications 
            WHERE application_status = 'Approved'
        """
        
        customer_df = read_sql_query(customer_query)
        bureau_df = read_sql_query(bureau_query)
        applications_df = read_sql_query(applications_query)
        
        logger.info(f"Loaded {len(customer_df)} customers, {len(applications_df)} applications")
        
        # Create features
        features_df = self.feature_engineer.create_credit_features(
            customer_df, bureau_df, applications_df
        )
        
        # Merge with application data for target variable
        model_df = features_df.merge(
            applications_df[['customer_id', 'default_flag']], 
            on='customer_id', 
            how='inner'
        )
        
        # Prepare feature matrix and target
        exclude_cols = [
            'customer_id', 'default_flag', 'application_id', 
            'last_activity_date', 'application_date'
        ]
        feature_cols = [col for col in model_df.columns if col not in exclude_cols]
        
        X = model_df[feature_cols]
        y = model_df['default_flag']
        
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
        logger.info(f"Default rate: {y.mean():.2%}")
        
        return X, y
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series, 
                   hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train a specific credit risk model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
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
        
        # Initialize model
        if hyperparameter_tuning:
            # Grid search for best parameters
            model = GridSearchCV(
                estimator=config['model'](**config['params']),
                param_grid=config['search_params'],
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
        else:
            model = config['model'](**config['params'])
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Get best model if grid search was used
        if hyperparameter_tuning:
            best_model = model.best_estimator_
            best_params = model.best_params_
            logger.info(f"Best parameters for {model_name}: {best_params}")
        else:
            best_model = model
            best_params = config['params']
        
        # Calibrate probabilities
        calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = self._evaluate_model(calibrated_model, X_val, y_val, model_name)
        
        # Store model
        self.models[model_name] = {
            'model': calibrated_model,
            'best_params': best_params,
            'metrics': metrics,
            'training_time': training_time,
            'feature_names': X_train.columns.tolist()
        }
        
        logger.info(f"Completed training {model_name} - AUC: {metrics['auc']:.4f}")
        return self.models[model_name]
    
    def _evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                       model_name: str) -> Dict[str, float]:
        """
        Evaluate model performance with credit risk metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Model name for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'gini': CreditRiskMetrics.calculate_gini_coefficient(y_test, y_pred_proba),
            'ks_statistic': CreditRiskMetrics.calculate_ks_statistic(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'accuracy': np.mean(y_pred == y_test)
        }
        
        # Log experiment to MLflow
        try:
            run_id = log_credit_risk_model(
                model=model,
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                hyperparameters=model.get_params(),
                metrics=metrics,
                feature_importance=feature_importance,
                model_type="sklearn"
            )
            logger.info(f"Credit risk experiment logged to MLflow: {run_id}")
        except Exception as e:
            logger.warning(f"Failed to log experiment to MLflow: {e}")
        
        return metrics
    
    def train_all_models(self, test_size: float = 0.2,
                        hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train all credit risk models and compare performance.
        
        Args:
            test_size: Proportion of data for testing
            hyperparameter_tuning: Whether to tune hyperparameters
            
        Returns:
            Dictionary with all trained models and comparison
        """
        logger.info("Starting credit risk model training pipeline...")
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Initialize feature transformer
        self.feature_transformer = FeatureTransformer(feature_type='credit')
        
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
        
        # Train all models
        results = {}
        for model_name in self.model_configs.keys():
            try:
                model_result = self.train_model(
                    model_name, X_train_transformed, y_train,
                    X_val_transformed, y_val, hyperparameter_tuning
                )
                results[model_name] = model_result
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Final evaluation on test set
        test_results = {}
        for model_name, model_info in results.items():
            test_metrics = self._evaluate_model(
                model_info['model'], X_test_transformed, y_test, f"{model_name}_test"
            )
            test_results[model_name] = test_metrics
        
        # Model comparison
        comparison_df = pd.DataFrame(test_results).T
        comparison_df = comparison_df.round(4)
        
        logger.info("Model Performance Comparison:")
        logger.info("\n" + comparison_df.to_string())
        
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
        """Save trained models to disk."""
        model_path = Path(self.settings.paths.model_store_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model_info in models.items():
            # Save model
            model_file = model_path / f"credit_{model_name}_{timestamp}.joblib"
            joblib.dump(model_info['model'], model_file)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'training_date': datetime.now().isoformat(),
                'metrics': model_info['metrics'],
                'best_params': model_info['best_params'],
                'feature_names': model_info['feature_names']
            }
            
            metadata_file = model_path / f"credit_{model_name}_{timestamp}_metadata.json"
            with open(metadata_file, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
        
        # Save feature transformer
        transformer_file = model_path / f"credit_feature_transformer_{timestamp}.joblib"
        joblib.dump(self.feature_transformer, transformer_file)
        
        logger.info(f"Saved models to {model_path}")


class CreditModelExplainer:
    """Model interpretability for credit risk models."""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = None
        self.lime_explainer = None
    
    def initialize_explainers(self, X_train: pd.DataFrame) -> None:
        """
        Initialize SHAP and LIME explainers.
        
        Args:
            X_train: Training data for explainer initialization
        """
        logger.info("Initializing model explainers...")
        
        # Initialize SHAP explainer
        try:
            if hasattr(self.model, 'predict_proba'):
                self.shap_explainer = shap.Explainer(self.model.predict_proba, X_train)
            else:
                self.shap_explainer = shap.Explainer(self.model, X_train)
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
        
        # Initialize LIME explainer
        try:
            self.lime_explainer = LimeTabularExplainer(
                X_train.values,
                feature_names=self.feature_names,
                class_names=['No Default', 'Default'],
                mode='classification',
                discretize_continuous=True
            )
        except Exception as e:
            logger.warning(f"Could not initialize LIME explainer: {e}")
    
    def explain_prediction(self, X_instance: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate explanation for a single prediction.
        
        Args:
            X_instance: Single instance to explain
            
        Returns:
            Dictionary with explanations
        """
        explanations = {}
        
        # Model prediction
        if hasattr(self.model, 'predict_proba'):
            prediction_proba = self.model.predict_proba(X_instance)[0]
            explanations['prediction_probability'] = {
                'no_default': float(prediction_proba[0]),
                'default': float(prediction_proba[1])
            }
        
        prediction = self.model.predict(X_instance)[0]
        explanations['prediction'] = int(prediction)
        
        # SHAP explanation
        if self.shap_explainer:
            try:
                shap_values = self.shap_explainer(X_instance)
                if hasattr(shap_values, 'values'):
                    shap_values_array = shap_values.values[0]
                    if len(shap_values_array.shape) > 1:
                        shap_values_array = shap_values_array[:, 1]  # Default class
                else:
                    shap_values_array = shap_values[0]
                
                shap_explanation = {
                    feature: float(value)
                    for feature, value in zip(self.feature_names, shap_values_array)
                }
                
                # Sort by absolute importance
                sorted_shap = sorted(
                    shap_explanation.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                
                explanations['shap_values'] = dict(sorted_shap[:10])  # Top 10 features
                
            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}")
                explanations['shap_values'] = {}
        
        # LIME explanation
        if self.lime_explainer:
            try:
                lime_explanation = self.lime_explainer.explain_instance(
                    X_instance.values[0],
                    self.model.predict_proba,
                    num_features=10
                )
                
                lime_values = {}
                for feature_idx, importance in lime_explanation.as_list():
                    if isinstance(feature_idx, str):
                        feature_name = feature_idx
                    else:
                        feature_name = self.feature_names[feature_idx]
                    lime_values[feature_name] = importance
                
                explanations['lime_explanation'] = lime_values
                
            except Exception as e:
                logger.error(f"LIME explanation failed: {e}")
                explanations['lime_explanation'] = {}
        
        # Generate text explanation
        explanations['text_explanation'] = self._generate_text_explanation(explanations)
        
        return explanations
    
    def _generate_text_explanation(self, explanations: Dict[str, Any]) -> str:
        """Generate human-readable explanation text."""
        text_parts = []
        
        # Prediction
        if 'prediction_probability' in explanations:
            default_prob = explanations['prediction_probability']['default']
            text_parts.append(f"Default probability: {default_prob:.2%}")
            
            if default_prob > 0.7:
                text_parts.append("HIGH RISK - Strong indicators of potential default")
            elif default_prob > 0.5:
                text_parts.append("MEDIUM RISK - Some concerning factors present")
            else:
                text_parts.append("LOW RISK - Good creditworthiness indicators")
        
        # Top risk factors
        if 'shap_values' in explanations:
            top_factors = list(explanations['shap_values'].items())[:3]
            if top_factors:
                text_parts.append("\nTop contributing factors:")
                for feature, value in top_factors:
                    direction = "increases" if value > 0 else "decreases"
                    text_parts.append(f"- {feature}: {direction} risk by {abs(value):.3f}")
        
        return "\n".join(text_parts)
    
    def generate_global_explanation(self, X_sample: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate global model explanation using a sample of data.
        
        Args:
            X_sample: Sample of data for global explanation
            
        Returns:
            Dictionary with global explanations
        """
        global_explanation = {}
        
        # Feature importance from SHAP
        if self.shap_explainer:
            try:
                shap_values = self.shap_explainer(X_sample)
                if hasattr(shap_values, 'values'):
                    shap_values_array = shap_values.values
                    if len(shap_values_array.shape) > 2:
                        shap_values_array = shap_values_array[:, :, 1]  # Default class
                else:
                    shap_values_array = shap_values
                
                # Calculate mean absolute SHAP values
                mean_shap = np.mean(np.abs(shap_values_array), axis=0)
                feature_importance = {
                    feature: float(importance)
                    for feature, importance in zip(self.feature_names, mean_shap)
                }
                
                # Sort by importance
                sorted_importance = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                global_explanation['feature_importance'] = dict(sorted_importance)
                
            except Exception as e:
                logger.error(f"Global SHAP explanation failed: {e}")
        
        return global_explanation


def train_credit_models(hyperparameter_tuning: bool = True) -> Dict[str, Any]:
    """
    Main function to train all credit risk models.
    
    Args:
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        Dictionary with training results
    """
    trainer = CreditRiskModelTrainer()
    results = trainer.train_all_models(hyperparameter_tuning=hyperparameter_tuning)
    
    logger.info("Credit risk model training completed successfully")
    return results


if __name__ == "__main__":
    # Run model training
    results = train_credit_models(hyperparameter_tuning=True)
    print("Training completed!")
    print(results['comparison'])
