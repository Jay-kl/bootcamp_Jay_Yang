# High-Frequency Trading Factor Prediction API

## Overview

This project productionizes machine learning models for predicting high-frequency trading factors. The system provides a RESTful API that serves two trained models:

1. **Regression Model**: Predicts initiative sell rates based on trading patterns
2. **Classification Model**: Classifies trading scenarios as high/low sell rate categories

## 🏗️ Project Structure

```
stage13_productization/
├── notebooks/
│   └── stage13_productization_homework-starter.ipynb  # Main analysis notebook
├── src/
│   └── utils.py                                       # Reusable utility functions
├── model/
│   ├── regression_model.pkl                          # Trained regression model
│   └── classification_model.pkl                      # Trained classification model
├── requirements.txt                                   # Python dependencies
└── README.md                                         # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository and navigate to the project directory
cd homework/stage13_productization/

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
# Start Jupyter Notebook
jupyter notebook notebooks/stage13_productization_homework-starter.ipynb
```

The notebook will:
- Load and preprocess high-frequency trading data
- Train and evaluate machine learning models
- Save models to the `model/` directory
- Start the Flask API server
- Test all API endpoints

### 3. API Usage

Once the Flask server is running (automatically started by the notebook), you can access:

- **Health Check**: `GET http://127.0.0.1:5000/health`
- **Model Info**: `GET http://127.0.0.1:5000/info`
- **Regression Prediction**: `POST http://127.0.0.1:5000/predict/regression`
- **Classification Prediction**: `POST http://127.0.0.1:5000/predict/classification`
- **Model Visualization**: `GET http://127.0.0.1:5000/plot/model_comparison`

## 📊 Models

### Regression Model
- **Purpose**: Predicts initiative sell rate (continuous value)
- **Algorithm**: Linear Regression
- **Features**: Initiative buy rate, large buy rate, large sell rate
- **Performance**: R² score and RMSE metrics provided

### Classification Model
- **Purpose**: Predicts high/low sell rate category (binary classification)
- **Algorithm**: Logistic Regression with StandardScaler preprocessing
- **Features**: Same as regression model
- **Performance**: Classification report with precision, recall, F1-score

## 🔌 API Reference

### POST Endpoints

#### Regression Prediction
```bash
curl -X POST http://127.0.0.1:5000/predict/regression \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.2, 0.3]}'
```

Response:
```json
{
  "prediction": 0.4523,
  "model_type": "regression",
  "feature_names": ["S_LI_INITIATIVEBUYRATE", "S_LI_LARGEBUYRATE", "S_LI_LARGESELLRATE"],
  "input_features": [0.5, 0.2, 0.3]
}
```

#### Classification Prediction
```bash
curl -X POST http://127.0.0.1:5000/predict/classification \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.2, 0.3]}'
```

Response:
```json
{
  "prediction": 1,
  "probability": 0.7234,
  "threshold": 0.4567,
  "interpretation": "High sell rate",
  "model_type": "classification",
  "feature_names": ["S_LI_INITIATIVEBUYRATE", "S_LI_LARGEBUYRATE", "S_LI_LARGESELLRATE"],
  "input_features": [0.5, 0.2, 0.3]
}
```

### GET Endpoints

#### Quick Predictions (URL Parameters)
```bash
# Regression
curl http://127.0.0.1:5000/predict/regression/0.5/0.2/0.3

# Classification  
curl http://127.0.0.1:5000/predict/classification/0.5/0.2/0.3
```

#### Health Check
```bash
curl http://127.0.0.1:5000/health
```

#### Model Information
```bash
curl http://127.0.0.1:5000/info
```

#### Model Visualization
Open in browser: `http://127.0.0.1:5000/plot/model_comparison`

## 📈 Features

The models use three high-frequency trading factors:

1. **S_LI_INITIATIVEBUYRATE**: Rate of initiative buy orders (0.0 - 1.0)
2. **S_LI_LARGEBUYRATE**: Rate of large buy orders (0.0 - 1.0)  
3. **S_LI_LARGESELLRATE**: Rate of large sell orders (0.0 - 1.0)

**Target Variable**: `S_LI_INITIATIVESELLRATE` (initiative sell rate)

## 🛠️ Development

### Utility Functions

The `src/utils.py` module provides reusable functions:

- `load_high_frequency_data()`: Data loading and preprocessing
- `prepare_model_data()`: Feature engineering
- `train_regression_model()` / `train_classification_model()`: Model training
- `save_model()` / `load_model()`: Model persistence
- `make_prediction()`: Prediction interface
- `validate_features()`: Input validation

### Adding New Models

1. Implement training function in `utils.py`
2. Save model using `save_model()`
3. Add new API endpoints in the Flask app
4. Update this README with new endpoint documentation

## 🧪 Testing

The notebook includes comprehensive API testing:

1. Health checks and model loading verification
2. Regression and classification predictions
3. GET and POST endpoint testing
4. Error handling validation
5. Model visualization testing

## 📋 Production Considerations

### Deployment Options

1. **Local Development**: Run notebook and Flask server locally
2. **Docker**: Create Dockerfile for containerized deployment
3. **Cloud**: Deploy to AWS, GCP, or Azure using services like:
   - AWS EC2 + Application Load Balancer
   - Google Cloud Run
   - Azure Container Instances

### Performance Optimizations

- Model caching for faster predictions
- Request batching for multiple predictions
- Async endpoints for non-blocking requests
- Model versioning for A/B testing

### Security

- Input validation and sanitization
- Rate limiting for API endpoints
- Authentication and authorization
- HTTPS in production

### Monitoring

- Prediction logging and metrics
- Model performance monitoring
- API uptime and latency tracking
- Error rate monitoring

## 🐛 Troubleshooting

### Common Issues

1. **Models not loading**: Ensure models are trained and saved in `model/` directory
2. **Import errors**: Check Python path and package installations
3. **Flask port conflicts**: Change port in `run_flask()` function
4. **Data file errors**: The system will create sample data if original files are missing

### Error Messages

- `Models not loaded`: Run the training cells in the notebook first
- `Invalid features`: Ensure feature values are numeric and between 0-1
- `Connection refused`: Flask server may not be running

## 📝 License

This project is part of the bootcamp coursework for educational purposes.

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include error handling
4. Update tests and documentation
5. Maintain backward compatibility

---

**Contact**: For questions about this implementation, refer to the notebook documentation or utility function docstrings.
