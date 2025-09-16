#!/usr/bin/env python3
"""
Backend API Server for Crop Yield Prediction
Flask-based REST API that serves the trained models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connections

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Classes (identical to training script)
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.skip_connection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
    def forward(self, x):
        residual = self.skip_connection(x)
        out = F.relu(self.fc1(x))
        out = self.bn1(out)
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class YieldPredictionModel(nn.Module):
    def __init__(self, input_dim):
        super(YieldPredictionModel, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        self.res_block1 = ResidualBlock(128, 128, 0.3)
        self.res_block2 = ResidualBlock(128, 64, 0.3)
        self.res_block3 = ResidualBlock(64, 32, 0.2)
        self.final_layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        x = self.initial_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.final_layers(x)
        return x

class RecommendationModel(nn.Module):
    def __init__(self, input_dim):
        super(RecommendationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fertilizer_head = nn.Linear(64, 3)
        self.irrigation_head = nn.Linear(64, 4)
        self.pest_control_head = nn.Linear(64, 3)
        
    def forward(self, x):
        x = self.encoder(x)
        fertilizer_output = F.softmax(self.fertilizer_head(x), dim=1)
        irrigation_output = F.softmax(self.irrigation_head(x), dim=1)
        pest_control_output = F.softmax(self.pest_control_head(x), dim=1)
        return fertilizer_output, irrigation_output, pest_control_output

class ModelServer:
    """Model server class to handle model loading and predictions"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.preprocessors = {}
        self.metadata = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models and preprocessors"""
        try:
            logger.info(f"Loading models from {self.models_dir}")
            
            # Load metadata
            with open(self.models_dir / "model_metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)
            
            # Load yield prediction model
            yield_model = YieldPredictionModel(self.metadata['model_info']['yield_model_input_dim']).to(device)
            yield_model.load_state_dict(torch.load(self.models_dir / "yield_model.pth", map_location=device))
            yield_model.eval()
            self.models['yield_model'] = yield_model
            
            # Load recommendation model
            rec_model = RecommendationModel(self.metadata['model_info']['recommendation_model_input_dim']).to(device)
            rec_model.load_state_dict(torch.load(self.models_dir / "recommendation_model.pth", map_location=device))
            rec_model.eval()
            self.models['recommendation_model'] = rec_model
            
            # Load preprocessors
            for name in ['scaler_yield', 'scaler_recommendation', 'crop_encoder', 'soil_encoder']:
                with open(self.models_dir / f"{name}.pkl", "rb") as f:
                    self.preprocessors[name] = pickle.load(f)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def predict_yield(self, input_data):
        """Predict crop yield from input data"""
        try:
            # Prepare features
            features = np.array([[
                self.preprocessors['crop_encoder'].transform([input_data['crop_type']])[0],
                self.preprocessors['soil_encoder'].transform([input_data['soil_type']])[0],
                input_data['soil_ph'],
                input_data['temperature'],
                input_data['humidity'],
                input_data['wind_speed'],
                input_data['nitrogen'],
                input_data['phosphorous'],
                input_data['potassium'],
                input_data['soil_quality']
            ]])
            
            # Scale features
            features_scaled = self.preprocessors['scaler_yield'].transform(features)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.models['yield_model'](torch.FloatTensor(features_scaled).to(device))
                yield_value = prediction.cpu().numpy()[0][0]
            
            return max(0.1, float(yield_value))
        
        except Exception as e:
            logger.error(f"Error predicting yield: {e}")
            raise

    def get_recommendations(self, input_data, current_yield):
        """Get farming recommendations based on input data and current yield"""
        try:
            # Prepare features with yield
            features = np.array([[
                self.preprocessors['crop_encoder'].transform([input_data['crop_type']])[0],
                self.preprocessors['soil_encoder'].transform([input_data['soil_type']])[0],
                input_data['soil_ph'],
                input_data['temperature'],
                input_data['humidity'],
                input_data['wind_speed'],
                input_data['nitrogen'],
                input_data['phosphorous'],
                input_data['potassium'],
                input_data['soil_quality'],
                current_yield
            ]])
            
            # Scale features
            features_scaled = self.preprocessors['scaler_recommendation'].transform(features)
            
            # Make predictions
            with torch.no_grad():
                f_out, i_out, p_out = self.models['recommendation_model'](torch.FloatTensor(features_scaled).to(device))
                f_rec = torch.argmax(f_out, 1).item()
                i_rec = torch.argmax(i_out, 1).item()
                p_rec = torch.argmax(p_out, 1).item()
            
            # Map recommendations
            fertilizer_options = ["Increase NPK", "Maintain NPK", "Reduce NPK"]
            irrigation_options = ["Intensive Irrigation", "Moderate Irrigation", "Reduced Irrigation", "Normal Irrigation"]
            pest_control_options = ["Intensive Pest Control", "Moderate Pest Control", "Minimal Pest Control"]
            
            return {
                'fertilizer': fertilizer_options[f_rec],
                'irrigation': irrigation_options[i_rec],
                'pest_control': pest_control_options[p_rec],
                'fertilizer_idx': f_rec,
                'irrigation_idx': i_rec,
                'pest_control_idx': p_rec
            }
        
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise

    def calculate_optimization(self, recommendations):
        """Calculate yield optimization based on recommendations"""
        improvement_factor = 1.0
        
        # Add improvements based on recommendations
        if recommendations['fertilizer_idx'] == 0:  # Increase NPK
            improvement_factor += 0.08
        if recommendations['irrigation_idx'] == 0:  # Intensive Irrigation
            improvement_factor += 0.06
        if recommendations['pest_control_idx'] == 0:  # Intensive Pest Control
            improvement_factor += 0.04
        
        return max(1.05, improvement_factor)

# Initialize model server
model_server = None

def initialize_models():
    """Initialize the model server"""
    global model_server
    try:
        model_server = ModelServer()
        return True
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        return False

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': model_server is not None
    })

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model metadata and available options"""
    if not model_server:
        return jsonify({'error': 'Models not loaded'}), 500
    
    return jsonify({
        'crop_types': model_server.metadata['model_info']['crop_classes'],
        'soil_types': model_server.metadata['model_info']['soil_classes'],
        'feature_names': model_server.metadata['feature_names'],
        'training_info': model_server.metadata['training_config']
    })

@app.route('/api/predict', methods=['POST'])
def predict_yield():
    """Main prediction endpoint"""
    try:
        if not model_server:
            return jsonify({'error': 'Models not loaded'}), 500
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'crop_type', 'soil_type', 'soil_ph', 'temperature', 'humidity',
            'wind_speed', 'nitrogen', 'phosphorous', 'potassium', 'soil_quality'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400
        
        # Make yield prediction
        current_yield = model_server.predict_yield(data)
        
        # Get recommendations
        recommendations = model_server.get_recommendations(data, current_yield)
        
        # Calculate optimization
        improvement_factor = model_server.calculate_optimization(recommendations)
        optimized_yield = current_yield * improvement_factor
        improvement_percentage = (improvement_factor - 1) * 100
        additional_production = optimized_yield - current_yield
        
        # Prepare response
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'input_data': data,
            'results': {
                'current_yield': round(current_yield, 2),
                'optimized_yield': round(optimized_yield, 2),
                'improvement_percentage': round(improvement_percentage, 1),
                'additional_production': round(additional_production, 2),
                'recommendations': recommendations
            }
        }
        
        logger.info(f"Prediction made for {data['crop_type']} - Yield: {current_yield:.2f}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple inputs"""
    try:
        if not model_server:
            return jsonify({'error': 'Models not loaded'}), 500
        
        data = request.get_json()
        inputs = data.get('inputs', [])
        
        if not inputs:
            return jsonify({'error': 'No inputs provided'}), 400
        
        results = []
        for i, input_data in enumerate(inputs):
            try:
                current_yield = model_server.predict_yield(input_data)
                recommendations = model_server.get_recommendations(input_data, current_yield)
                improvement_factor = model_server.calculate_optimization(recommendations)
                optimized_yield = current_yield * improvement_factor
                
                results.append({
                    'index': i,
                    'input': input_data,
                    'current_yield': round(current_yield, 2),
                    'optimized_yield': round(optimized_yield, 2),
                    'recommendations': recommendations
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main function to start the server"""
    logger.info("Starting Crop Yield Prediction API Server...")
    
    # Initialize models
    if not initialize_models():
        logger.error("Failed to load models. Exiting.")
        return
    
    logger.info("Models loaded successfully")
    logger.info("Starting Flask server...")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )

if __name__ == "__main__":
    main()