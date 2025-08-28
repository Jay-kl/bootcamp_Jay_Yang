#!/usr/bin/env python3
"""
High-Frequency Trading Factor Prediction API

Standalone Flask application for serving trained machine learning models
that predict high-frequency trading factors.

Usage:
    python app.py

This will start the API server on http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.append('src')
from utils import load_model, make_prediction

app = Flask(__name__)

# Load models on startup
model_dir = 'model/'
regression_model = None
classification_model = None

def load_models():
    """Load trained models from disk"""
    global regression_model, classification_model
    
    try:
        regression_model = load_model(os.path.join(model_dir, 'regression_model.pkl'))
        classification_model = load_model(os.path.join(model_dir, 'classification_model.pkl'))
        print("✓ Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run the notebook first to train and save models.")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_loaded = regression_model is not None and classification_model is not None
    return jsonify({
        'status': 'healthy' if models_loaded else 'error',
        'models_loaded': models_loaded,
        'message': 'Trading Factor Prediction API is running'
    })

@app.route('/predict/regression', methods=['POST'])
def predict_regression():
    """Predict initiative sell rate using regression model"""
    try:
        data = request.get_json()
        features = data.get('features', None)
        
        if features is None:
            return jsonify({'error': 'No features provided'}), 400
        
        if regression_model is None:
            return jsonify({'error': 'Regression model not loaded'}), 500
            
        result = make_prediction(regression_model, features)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/classification', methods=['POST'])
def predict_classification():
    """Predict high/low sell rate category using classification model"""
    try:
        data = request.get_json()
        features = data.get('features', None)
        
        if features is None:
            return jsonify({'error': 'No features provided'}), 400
            
        if classification_model is None:
            return jsonify({'error': 'Classification model not loaded'}), 500
            
        result = make_prediction(classification_model, features)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/regression/<float:buy_rate>/<float:large_buy>/<float:large_sell>', methods=['GET'])
def predict_regression_get(buy_rate, large_buy, large_sell):
    """GET endpoint for regression prediction with URL parameters"""
    try:
        features = [buy_rate, large_buy, large_sell]
        
        if regression_model is None:
            return jsonify({'error': 'Regression model not loaded'}), 500
            
        result = make_prediction(regression_model, features)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/classification/<float:buy_rate>/<float:large_buy>/<float:large_sell>', methods=['GET'])
def predict_classification_get(buy_rate, large_buy, large_sell):
    """GET endpoint for classification prediction with URL parameters"""
    try:
        features = [buy_rate, large_buy, large_sell]
        
        if classification_model is None:
            return jsonify({'error': 'Classification model not loaded'}), 500
            
        result = make_prediction(classification_model, features)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/plot/model_comparison')
def plot_model_comparison():
    """Generate visualization comparing model predictions"""
    try:
        if regression_model is None or classification_model is None:
            return jsonify({'error': 'Models not loaded'}), 500
            
        # Create sample predictions for visualization
        buy_rates = np.linspace(0.3, 0.7, 10)
        reg_preds = []
        clf_preds = []
        
        for buy_rate in buy_rates:
            features = [buy_rate, 0.2, 0.3]  # Fixed large buy/sell rates
            
            reg_result = make_prediction(regression_model, features)
            clf_result = make_prediction(classification_model, features)
            
            reg_preds.append(reg_result['prediction'])
            clf_preds.append(clf_result['probability'])
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Regression predictions
        ax1.plot(buy_rates, reg_preds, 'b-o', label='Predicted Sell Rate')
        ax1.set_xlabel('Initiative Buy Rate')
        ax1.set_ylabel('Predicted Sell Rate')
        ax1.set_title('Regression Model Predictions')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Classification probabilities
        ax2.plot(buy_rates, clf_preds, 'r-s', label='High Sell Rate Probability')
        ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Decision Threshold')
        ax2.set_xlabel('Initiative Buy Rate')
        ax2.set_ylabel('Probability')
        ax2.set_title('Classification Model Predictions')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_bytes = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return f'<img src="data:image/png;base64,{img_bytes}" style="max-width:100%;height:auto;"/>'
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    info = {
        'regression_model': {
            'loaded': regression_model is not None,
            'features': regression_model['feature_names'] if regression_model else None,
            'model_type': 'Linear Regression'
        },
        'classification_model': {
            'loaded': classification_model is not None,
            'features': classification_model['feature_names'] if classification_model else None,
            'model_type': 'Logistic Regression Pipeline',
            'threshold': classification_model['threshold'] if classification_model else None
        },
        'endpoints': [
            'GET /health - Health check',
            'POST /predict/regression - Regression prediction', 
            'POST /predict/classification - Classification prediction',
            'GET /predict/regression/<buy_rate>/<large_buy>/<large_sell> - Quick regression',
            'GET /predict/classification/<buy_rate>/<large_buy>/<large_sell> - Quick classification',
            'GET /plot/model_comparison - Model visualization',
            'GET /info - This information'
        ]
    }
    return jsonify(info)

@app.route('/')
def index():
    """Landing page with API documentation"""
    return """
    <html>
    <head><title>Trading Factor Prediction API</title></head>
    <body>
        <h1>🚀 High-Frequency Trading Factor Prediction API</h1>
        <p>This API provides machine learning predictions for high-frequency trading factors.</p>
        
        <h2>📊 Available Endpoints:</h2>
        <ul>
            <li><a href="/health">GET /health</a> - Health check</li>
            <li><a href="/info">GET /info</a> - Model information</li>
            <li><a href="/plot/model_comparison">GET /plot/model_comparison</a> - Model visualization</li>
            <li>POST /predict/regression - Regression prediction</li>
            <li>POST /predict/classification - Classification prediction</li>
        </ul>
        
        <h2>🧪 Quick Test:</h2>
        <p>Try these GET endpoints with sample data:</p>
        <ul>
            <li><a href="/predict/regression/0.5/0.2/0.3">Regression: 0.5, 0.2, 0.3</a></li>
            <li><a href="/predict/classification/0.6/0.25/0.35">Classification: 0.6, 0.25, 0.35</a></li>
        </ul>
        
        <h2>📝 Example POST Request:</h2>
        <pre>
curl -X POST http://127.0.0.1:5000/predict/regression \\
  -H "Content-Type: application/json" \\
  -d '{"features": [0.5, 0.2, 0.3]}'
        </pre>
        
        <p><strong>Features:</strong> [initiative_buy_rate, large_buy_rate, large_sell_rate]</p>
        <p>All values should be between 0.0 and 1.0</p>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("🚀 Starting High-Frequency Trading Factor Prediction API...")
    
    # Load models
    if load_models():
        print("📊 Available endpoints:")
        print("   - http://127.0.0.1:5000/ (Landing page)")
        print("   - http://127.0.0.1:5000/health")
        print("   - http://127.0.0.1:5000/info")
        print("   - http://127.0.0.1:5000/predict/regression (POST)")
        print("   - http://127.0.0.1:5000/predict/classification (POST)")
        print("   - http://127.0.0.1:5000/plot/model_comparison")
        print("\n✨ Ready to serve predictions!")
        
        app.run(host='127.0.0.1', port=5000, debug=False)
    else:
        print("❌ Failed to load models. Please run the notebook first to train models.")
        print("   1. Open: notebooks/stage13_productization_homework-starter.ipynb")
        print("   2. Run all cells to train and save models")
        print("   3. Then run: python app.py")
