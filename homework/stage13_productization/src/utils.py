"""
Utility functions for high-frequency trading factor analysis and model deployment.

This module contains reusable functions for data processing, model training,
and prediction serving for the high-frequency trading factor models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, classification_report
import pickle
import os
from typing import Tuple, Dict, Any, List


def load_high_frequency_data(data_path: str) -> pd.DataFrame:
    """
    Load and combine high-frequency trading data from CSV files.
    
    Args:
        data_path: Path to directory containing data files
        
    Returns:
        Combined DataFrame with high-frequency trading data
    """
    try:
        df1 = pd.read_csv(os.path.join(data_path, "high_frequency_data1.csv"))
        df2 = pd.read_csv(os.path.join(data_path, "high_frequency_data2.csv"))
        return pd.concat([df1, df2], ignore_index=True)
    except FileNotFoundError:
        # Create sample data if files not found
        np.random.seed(42)
        n_samples = 1000
        return pd.DataFrame({
            'S_LI_INITIATIVEBUYRATE': np.random.uniform(0.3, 0.7, n_samples),
            'S_LI_INITIATIVESELLRATE': np.random.uniform(0.3, 0.7, n_samples),
            'S_LI_LARGEBUYRATE': np.random.uniform(0.1, 0.3, n_samples),
            'S_LI_LARGESELLRATE': np.random.uniform(0.1, 0.3, n_samples),
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1min')
        })


def prepare_model_data(data_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target variables for model training.
    
    Args:
        data_df: Raw high-frequency trading data
        
    Returns:
        Tuple of (features_df, target_series)
    """
    feature_cols = [
        'S_LI_INITIATIVEBUYRATE',
        'S_LI_LARGEBUYRATE', 
        'S_LI_LARGESELLRATE'
    ]
    target_col = 'S_LI_INITIATIVESELLRATE'
    
    # Clean data
    df_clean = data_df[feature_cols + [target_col]].dropna()
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    return X, y


def train_regression_model(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Train linear regression model for sell rate prediction.
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        Dictionary containing trained model and performance metrics
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return {
        'model': model,
        'r2_score': r2,
        'rmse': rmse,
        'feature_names': list(X.columns),
        'model_type': 'regression'
    }


def train_classification_model(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Train classification model for high/low sell rate prediction.
    
    Args:
        X: Feature matrix
        y: Target variable (will be converted to binary)
        
    Returns:
        Dictionary containing trained model and performance metrics
    """
    # Create binary target
    y_binary = (y > y.median()).astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )
    
    # Train pipeline with scaling
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    return {
        'model': model,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_names': list(X.columns),
        'threshold': float(y.median()),
        'model_type': 'classification'
    }


def save_model(model_dict: Dict[str, Any], model_path: str) -> None:
    """
    Save trained model to pickle file.
    
    Args:
        model_dict: Dictionary containing model and metadata
        model_path: Path to save the model file
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)


def load_model(model_path: str) -> Dict[str, Any]:
    """
    Load trained model from pickle file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model and metadata
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def validate_features(features: List[float], expected_feature_names: List[str]) -> bool:
    """
    Validate input features for prediction.
    
    Args:
        features: List of feature values
        expected_feature_names: Expected feature names from model
        
    Returns:
        True if valid, False otherwise
    """
    if len(features) != len(expected_feature_names):
        return False
    
    # Check if all values are numeric and in reasonable range
    for feature in features:
        if not isinstance(feature, (int, float)):
            return False
        if not (0 <= feature <= 1):  # Trading rates should be between 0 and 1
            return False
    
    return True


def make_prediction(model_dict: Dict[str, Any], features: List[float]) -> Dict[str, Any]:
    """
    Make prediction using trained model.
    
    Args:
        model_dict: Dictionary containing model and metadata
        features: List of feature values
        
    Returns:
        Dictionary containing prediction and metadata
    """
    model = model_dict['model']
    feature_names = model_dict['feature_names']
    
    # Validate input
    if not validate_features(features, feature_names):
        raise ValueError(f"Invalid features. Expected {len(feature_names)} values between 0 and 1.")
    
    # Convert to DataFrame for prediction
    X_pred = pd.DataFrame([features], columns=feature_names)
    
    if model_dict['model_type'] == 'regression':
        prediction = float(model.predict(X_pred)[0])
        return {
            'prediction': prediction,
            'model_type': 'regression',
            'feature_names': feature_names,
            'input_features': features
        }
    else:  # classification
        prediction = int(model.predict(X_pred)[0])
        probability = float(model.predict_proba(X_pred)[0][1])
        threshold = model_dict['threshold']
        
        return {
            'prediction': prediction,
            'probability': probability,
            'threshold': threshold,
            'interpretation': 'High sell rate' if prediction == 1 else 'Low sell rate',
            'model_type': 'classification',
            'feature_names': feature_names,
            'input_features': features
        }


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate summary statistics for a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of summary statistics
    """
    return {
        'mean': float(df.mean().mean()),
        'std': float(df.std().mean()),
        'min': float(df.min().min()),
        'max': float(df.max().max()),
        'count': int(df.count().sum())
    }
