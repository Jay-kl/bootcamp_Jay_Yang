#!/usr/bin/env python3
"""
High-Frequency Trading Factor Prediction API
Stage 13 Production Deployment Application

A production-ready Flask API for serving high-frequency trading factor prediction models.
Supports both regression and classification predictions with comprehensive error handling,
performance monitoring, and visualization capabilities.
"""

import os
import sys
import pickle
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server deployment
import matplotlib.pyplot as plt
import io
import base64

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Set up production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trading_factor_api')

# Initialize Flask application
app = Flask(__name__)

# Global model storage
MODELS = {
    'regression': None,
    'classification': None,
    'loaded': False
}

# Model metadata
MODEL_METADATA = {
    'service_name': 'High-Frequency Trading Factor Prediction API',
    'version': '1.0.0',
    'features': ['S_LI_INITIATIVEBUYRATE', 'S_LI_LARGEBUYRATE', 'S_LI_LARGESELLRATE'],
    'target': 'S_LI_INITIATIVESELLRATE'
}


def load_models(model_dir: str = '../models/') -> bool:
    """Load production models from pickle files."""
    try:
        regression_path = os.path.join(model_dir, 'regression_model.pkl')
        classification_path = os.path.join(model_dir, 'classification_model.pkl')
        
        with open(regression_path, 'rb') as f:
            MODELS['regression'] = pickle.load(f)
        
        with open(classification_path, 'rb') as f:
            MODELS['classification'] = pickle.load(f)
        
        MODELS['loaded'] = True
        logger.info("✅ Models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        MODELS['loaded'] = False
        return False


def validate_features(features: List[float]) -> tuple[bool, str]:
    """Validate input features for prediction."""
    if len(features) != 3:
        return False, f"Expected 3 features, got {len(features)}"
    
    for i, feature in enumerate(features):
        if not isinstance(feature, (int, float)):
            return False, f"Feature {i} is not numeric: {feature}"
        if not (0 <= feature <= 1):
            logger.warning(f"Feature {i} outside typical range [0,1]: {feature}")
    
    return True, "Valid"


def make_prediction(model_type: str, features: List[float]) -> Dict[str, Any]:
    """Make a prediction with the specified model."""
    if not MODELS['loaded']:
        raise RuntimeError("Models not loaded")
    
    # Validate input
    is_valid, error_msg = validate_features(features)
    if not is_valid:
        raise ValueError(error_msg)
    
    model_dict = MODELS[model_type]
    X_pred = pd.DataFrame([features], columns=model_dict['feature_names'])
    
    if model_type == 'regression':
        prediction = float(model_dict['model'].predict(X_pred)[0])
        
        return {
            'prediction': prediction,
            'model_type': 'regression',
            'feature_names': model_dict['feature_names'],
            'input_features': features,
            'confidence': 'high' if 0.3 <= prediction <= 0.7 else 'medium',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    elif model_type == 'classification':
        prediction = int(model_dict['model'].predict(X_pred)[0])
        probability = float(model_dict['model'].predict_proba(X_pred)[0][1])
        
        return {
            'prediction': prediction,
            'probability': probability,
            'threshold': model_dict['threshold'],
            'interpretation': 'High sell rate' if prediction == 1 else 'Low sell rate',
            'model_type': 'classification',
            'feature_names': model_dict['feature_names'],
            'input_features': features,
            'confidence': 'high' if abs(probability - 0.5) > 0.3 else 'medium',
            'timestamp': datetime.utcnow().isoformat()
        }


# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return jsonify({
        'status': 'healthy' if MODELS['loaded'] else 'unhealthy',
        'models_loaded': MODELS['loaded'],
        'timestamp': datetime.utcnow().isoformat(),
        'service': MODEL_METADATA['service_name'],
        'version': MODEL_METADATA['version']
    })


@app.route('/info', methods=['GET'])
def model_info():
    """Comprehensive model and API information."""
    if not MODELS['loaded']:
        return jsonify({'error': 'Models not loaded'}), 500
    
    return jsonify({
        'service': MODEL_METADATA,
        'regression_model': {
            'type': MODELS['regression']['model_type'],
            'performance': MODELS['regression']['performance'],
            'trained_at': MODELS['regression']['metadata']['trained_at']
        },
        'classification_model': {
            'type': MODELS['classification']['model_type'],
            'threshold': MODELS['classification']['threshold'],
            'performance': MODELS['classification']['performance'],
            'trained_at': MODELS['classification']['metadata']['trained_at']
        },
        'api_endpoints': [
            'GET /health - Health check',
            'GET /info - Model information',
            'POST /predict/regression - Regression prediction',
            'POST /predict/classification - Classification prediction',
            'GET /predict/regression/<buy_rate>/<large_buy>/<large_sell>',
            'GET /predict/classification/<buy_rate>/<large_buy>/<large_sell>',
            'GET /plot/model_comparison - Visualization'
        ]
    })


@app.route('/predict/regression', methods=['POST'])
def predict_regression():
    """Regression prediction endpoint."""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Missing features. Expected: {"features": [buy_rate, large_buy, large_sell]}'
            }), 400
        
        result = make_prediction('regression', data['features'])
        result['latency_ms'] = (time.time() - start_time) * 1000
        
        logger.info(f"Regression: {result['prediction']:.4f} ({result['latency_ms']:.2f}ms)")
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Regression error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/predict/classification', methods=['POST'])
def predict_classification():
    """Classification prediction endpoint."""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Missing features. Expected: {"features": [buy_rate, large_buy, large_sell]}'
            }), 400
        
        result = make_prediction('classification', data['features'])
        result['latency_ms'] = (time.time() - start_time) * 1000
        
        logger.info(f"Classification: {result['interpretation']} ({result['latency_ms']:.2f}ms)")
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/predict/regression/<float:buy_rate>/<float:large_buy>/<float:large_sell>')
def predict_regression_get(buy_rate, large_buy, large_sell):
    """Quick regression prediction via URL parameters."""
    try:
        result = make_prediction('regression', [buy_rate, large_buy, large_sell])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/classification/<float:buy_rate>/<float:large_buy>/<float:large_sell>')
def predict_classification_get(buy_rate, large_buy, large_sell):
    """Quick classification prediction via URL parameters."""
    try:
        result = make_prediction('classification', [buy_rate, large_buy, large_sell])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/plot/model_comparison')
def plot_model_comparison():
    """Generate model comparison visualization."""
    try:
        if not MODELS['loaded']:
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Generate sample predictions
        buy_rates = np.linspace(0.3, 0.7, 10)
        reg_preds, clf_probs = [], []
        
        for buy_rate in buy_rates:
            features = [buy_rate, 0.2, 0.3]
            reg_result = make_prediction('regression', features)
            clf_result = make_prediction('classification', features)
            
            reg_preds.append(reg_result['prediction'])
            clf_probs.append(clf_result['probability'])
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(buy_rates, reg_preds, 'b-o', linewidth=2)
        ax1.set_xlabel('Initiative Buy Rate')
        ax1.set_ylabel('Predicted Sell Rate')
        ax1.set_title('Regression Predictions')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(buy_rates, clf_probs, 'r-s', linewidth=2)
        ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Initiative Buy Rate')
        ax2.set_ylabel('High Sell Rate Probability')
        ax2.set_title('Classification Predictions')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        html = f'''
        <html>
        <body>
            <h2>High-Frequency Trading Factor Model Comparison</h2>
            <p>Real-time model predictions across different initiative buy rates</p>
            <img src="data:image/png;base64,{img_data}" style="max-width:100%;height:auto;"/>
            <p><strong>Generated:</strong> {datetime.utcnow().isoformat()}</p>
        </body>
        </html>
        '''
        return html
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load models on startup
    models_loaded = load_models()
    
    if models_loaded:
        logger.info("🚀 Starting High-Frequency Trading Factor Prediction API")
        logger.info("📊 Available endpoints:")
        logger.info("   - http://127.0.0.1:5000/health")
        logger.info("   - http://127.0.0.1:5000/info")
        logger.info("   - http://127.0.0.1:5000/predict/regression")
        logger.info("   - http://127.0.0.1:5000/predict/classification")
        logger.info("   - http://127.0.0.1:5000/plot/model_comparison")
        
        # Run in development mode
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    else:
        logger.error("❌ Cannot start server - model loading failed")
        sys.exit(1)
