#!/usr/bin/env python3
"""
Model Training Script for Crop Yield Prediction
Focuses only on training models and saving artifacts for backend deployment
"""

import os
import sys
import json
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score

# Configuration
class Config:
    BASE_DIR = Path.cwd()
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data" 
    LOGS_DIR = BASE_DIR / "logs"
    
    # Training Parameters
    YIELD_EPOCHS = 100
    REC_EPOCHS = 80
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0008
    TEST_SIZE = 0.15
    SYNTHETIC_SAMPLES = 15000
    
    @classmethod
    def create_directories(cls):
        for dir_attr in ['MODELS_DIR', 'DATA_DIR', 'LOGS_DIR']:
            directory = getattr(cls, dir_attr)
            directory.mkdir(parents=True, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model Classes (same as original)
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

class ModelTrainer:
    def __init__(self, config=None):
        self.config = config or Config()
        self.models = {}
        self.preprocessors = {}
        self.config.create_directories()
        
        self.feature_names = ['Crop_Type', 'Soil_Type', 'Soil_pH', 'Temperature', 'Humidity', 
                              'Wind_Speed', 'N', 'P', 'K', 'Soil_Quality']
    
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")
    
    def generate_data(self, n_samples=None):
        n_samples = n_samples or self.config.SYNTHETIC_SAMPLES
        self.log(f"Generating {n_samples} synthetic data samples...")
        
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=x) for x in range(n_samples)]
        
        crop_types = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Soybean', 'Barley']
        soil_types = ['Clay', 'Sandy', 'Loamy', 'Silt', 'Peat', 'Chalk']
        
        np.random.seed(42)
        
        data = {
            'Date': dates,
            'Crop_Type': np.random.choice(crop_types, n_samples),
            'Soil_Type': np.random.choice(soil_types, n_samples),
            'Soil_pH': np.clip(np.random.normal(6.5, 1.2, n_samples), 4.0, 8.5),
            'Temperature': np.clip(np.random.normal(25, 8, n_samples), 10, 45),
            'Humidity': np.clip(np.random.normal(65, 15, n_samples), 20, 95),
            'Wind_Speed': np.clip(np.random.exponential(3, n_samples), 0.5, 15),
            'N': np.clip(np.random.normal(120, 40, n_samples), 20, 300),
            'P': np.clip(np.random.normal(50, 20, n_samples), 10, 120),
            'K': np.clip(np.random.normal(80, 30, n_samples), 15, 200),
            'Soil_Quality': np.random.randint(1, 11, n_samples)
        }
        
        df = pd.DataFrame(data)
        df['Crop_Yield'] = self._calculate_yields(df)
        
        data_file = self.config.DATA_DIR / 'agricultural_data.csv'
        df.to_csv(data_file, index=False)
        
        self.log(f"Generated and saved data to {data_file}")
        return df
    
    def _calculate_yields(self, df):
        base_yields = {'Rice': 4.5, 'Wheat': 3.2, 'Maize': 5.8, 'Cotton': 2.1, 'Sugarcane': 45.0, 'Soybean': 2.8, 'Barley': 2.9}
        soil_multipliers = {'Clay': 0.95, 'Sandy': 0.85, 'Loamy': 1.1, 'Silt': 1.0, 'Peat': 0.9, 'Chalk': 0.8}
        temp_optimal = {'Rice': 28, 'Wheat': 20, 'Maize': 25, 'Cotton': 30, 'Sugarcane': 32, 'Soybean': 25, 'Barley': 18}
        
        yields = []
        for _, row in df.iterrows():
            base_yield = base_yields[row['Crop_Type']]
            soil_mult = soil_multipliers[row['Soil_Type']]
            ph_factor = max(0.3, 1 - abs(row['Soil_pH'] - 6.75) * 0.1)
            temp_factor = max(0.2, 1 - abs(row['Temperature'] - temp_optimal[row['Crop_Type']]) * 0.02)
            humidity_factor = max(0.4, 1 - abs(row['Humidity'] - 60) * 0.005)
            wind_factor = max(0.5, 1 - abs(row['Wind_Speed'] - 4) * 0.03)
            n_factor = min(1.2, row['N'] / 100)
            p_factor = min(1.15, row['P'] / 40)
            k_factor = min(1.1, row['K'] / 60)
            quality_factor = row['Soil_Quality'] / 10
            
            final_yield = (base_yield * soil_mult * ph_factor * temp_factor * humidity_factor * wind_factor * n_factor * p_factor * k_factor * quality_factor)
            final_yield += np.random.normal(0, final_yield * 0.1)
            yields.append(max(0.1, final_yield))
        return yields

    def preprocess_data(self, df):
        self.log("Preprocessing data...")
        
        X = df[self.feature_names].copy()
        y = df['Crop_Yield'].copy()
        
        self.preprocessors['crop_encoder'] = LabelEncoder()
        self.preprocessors['soil_encoder'] = LabelEncoder()
        
        X.loc[:, 'Crop_Type'] = self.preprocessors['crop_encoder'].fit_transform(X['Crop_Type'])
        X.loc[:, 'Soil_Type'] = self.preprocessors['soil_encoder'].fit_transform(X['Soil_Type'])
        
        self.preprocessors['scaler_yield'] = StandardScaler()
        X_scaled = self.preprocessors['scaler_yield'].fit_transform(X)
        
        self.log(f"Preprocessed data - Features: {X_scaled.shape}, Target: {y.shape}")
        return X_scaled, y.values, X
    
    def train_yield_model(self, X_train, y_train, X_val, y_val):
        self.log("Training yield prediction model...")
        
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        model = YieldPredictionModel(X_train.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.YIELD_EPOCHS):
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_yield_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    self.log(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 20 == 0:
                self.log(f"Epoch {epoch+1}/{self.config.YIELD_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        model.load_state_dict(torch.load('best_yield_model.pth'))
        os.remove('best_yield_model.pth')
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor).cpu().numpy()
        
        val_r2 = r2_score(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        self.models['yield_model'] = model
        self.log(f"Yield model trained - Val R²: {val_r2:.4f}, Val MAE: {val_mae:.4f}")
        return model, {'val_r2': float(val_r2), 'val_mae': float(val_mae)}

    def prepare_recommendation_data(self, X_original, X_scaled):
        yield_model = self.models['yield_model']
        yield_model.eval()
        
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        with torch.no_grad():
            predicted_yields = yield_model(X_tensor).cpu().numpy()
        
        X_rec = np.hstack([X_original.values, predicted_yields])
        
        self.preprocessors['scaler_recommendation'] = StandardScaler()
        X_rec_scaled = self.preprocessors['scaler_recommendation'].fit_transform(X_rec)
        
        n_samples = len(X_rec)
        fertilizer_targets, irrigation_targets, pest_control_targets = [], [], []
        
        for i in range(n_samples):
            n_level, humidity, temp = X_original.iloc[i][['N', 'Humidity', 'Temperature']]
            
            if n_level < 80: fertilizer_targets.append(0)
            elif n_level > 150: fertilizer_targets.append(2)
            else: fertilizer_targets.append(1)
            
            if humidity < 40 or temp > 35: irrigation_targets.append(0)
            elif humidity < 60: irrigation_targets.append(1)
            elif humidity > 80: irrigation_targets.append(2)
            else: irrigation_targets.append(3)
            
            if temp > 30 and humidity > 70: pest_control_targets.append(0)
            elif temp > 25: pest_control_targets.append(1)
            else: pest_control_targets.append(2)
            
        return X_rec_scaled, np.array(fertilizer_targets), np.array(irrigation_targets), np.array(pest_control_targets)

    def train_recommendation_model(self, X_original, X_scaled):
        self.log("Training recommendation model...")
        
        X_rec, fert_targets, irr_targets, pest_targets = self.prepare_recommendation_data(X_original, X_scaled)
        
        X_train, X_val, yf_train, yf_val, yi_train, yi_val, yp_train, yp_val = train_test_split(
            X_rec, fert_targets, irr_targets, pest_targets, test_size=self.config.TEST_SIZE, random_state=42)
        
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        yf_train_tensor = torch.LongTensor(yf_train).to(device)
        yi_train_tensor = torch.LongTensor(yi_train).to(device)
        yp_train_tensor = torch.LongTensor(yp_train).to(device)
        yf_val_tensor = torch.LongTensor(yf_val).to(device)
        yi_val_tensor = torch.LongTensor(yi_val).to(device)
        yp_val_tensor = torch.LongTensor(yp_val).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, yf_train_tensor, yi_train_tensor, yp_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, yf_val_tensor, yi_val_tensor, yp_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        model = RecommendationModel(X_train.shape[1]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.REC_EPOCHS):
            model.train()
            train_loss = 0.0
            for batch_X, bf, bi, bp in train_loader:
                optimizer.zero_grad()
                f_out, i_out, p_out = model(batch_X)
                loss = criterion(f_out, bf) + criterion(i_out, bi) + criterion(p_out, bp)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, bf, bi, bp in val_loader:
                    f_out, i_out, p_out = model(batch_X)
                    loss = criterion(f_out, bf) + criterion(i_out, bi) + criterion(p_out, bp)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_rec_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    self.log(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 15 == 0:
                self.log(f"Epoch {epoch+1}/{self.config.REC_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
        model.load_state_dict(torch.load('best_rec_model.pth'))
        os.remove('best_rec_model.pth')
        
        model.eval()
        with torch.no_grad():
            f_out, i_out, p_out = model(X_val_tensor)
            f_pred = torch.argmax(f_out, 1)
            i_pred = torch.argmax(i_out, 1) 
            p_pred = torch.argmax(p_out, 1)
        
        f_acc = accuracy_score(yf_val, f_pred.cpu())
        i_acc = accuracy_score(yi_val, i_pred.cpu())
        p_acc = accuracy_score(yp_val, p_pred.cpu())
        
        self.models['recommendation_model'] = model
        self.log(f"Recommendation model trained - Accuracies: F:{f_acc:.3f}, I:{i_acc:.3f}, P:{p_acc:.3f}")
        return model, {'fertilizer_accuracy': f_acc, 'irrigation_accuracy': i_acc, 'pest_control_accuracy': p_acc}

    def save_models(self):
        self.log("Saving all models and artifacts...")
        
        # Save PyTorch models
        torch.save(self.models['yield_model'].state_dict(), self.config.MODELS_DIR / 'yield_model.pth')
        torch.save(self.models['recommendation_model'].state_dict(), self.config.MODELS_DIR / 'recommendation_model.pth')
        
        # Save preprocessors
        for name, processor in self.preprocessors.items():
            with open(self.config.MODELS_DIR / f'{name}.pkl', 'wb') as f:
                pickle.dump(processor, f)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'model_info': {
                'yield_model_input_dim': len(self.feature_names),
                'recommendation_model_input_dim': len(self.feature_names) + 1,
                'crop_classes': list(self.preprocessors['crop_encoder'].classes_),
                'soil_classes': list(self.preprocessors['soil_encoder'].classes_),
            },
            'training_config': {
                'yield_epochs': self.config.YIELD_EPOCHS,
                'rec_epochs': self.config.REC_EPOCHS,
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE
            },
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.config.MODELS_DIR / 'model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
            
        self.log("All artifacts saved successfully")
        return True

    def test_models(self):
        self.log("Testing complete system...")
        try:
            # Test sample
            test_sample = {
                'Crop_Type': 'Rice', 'Soil_Type': 'Loamy', 'Soil_pH': 6.8, 'Temperature': 28, 
                'Humidity': 75, 'Wind_Speed': 3.2, 'N': 120, 'P': 50, 'K': 80, 'Soil_Quality': 8
            }
            
            features = np.array([[
                self.preprocessors['crop_encoder'].transform([test_sample['Crop_Type']])[0],
                self.preprocessors['soil_encoder'].transform([test_sample['Soil_Type']])[0],
                *list(test_sample.values())[2:]
            ]])
            
            features_scaled = self.preprocessors['scaler_yield'].transform(features)
            
            self.models['yield_model'].eval()
            with torch.no_grad():
                predicted_yield = self.models['yield_model'](torch.FloatTensor(features_scaled).to(device)).item()
            
            self.log(f"System test passed - Predicted yield for Rice: {predicted_yield:.2f}")
            return True
        except Exception as e:
            self.log(f"System testing failed: {e}")
            return False

    def train_complete_pipeline(self, data_path=None):
        self.log("Starting Complete Model Training Pipeline")
        start_time = time.time()
        
        try:
            # Stage 1: Data Generation/Loading
            self.log("="*50 + "\nSTAGE 1: DATA PREPARATION\n" + "="*50)
            if data_path and Path(data_path).exists():
                df = pd.read_csv(data_path)
                self.log(f"Loaded data from {data_path}")
            else:
                df = self.generate_data()

            # Stage 2: Training
            self.log("="*50 + "\nSTAGE 2: MODEL TRAINING\n" + "="*50)
            X_scaled, y, X_original = self.preprocess_data(df)
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=self.config.TEST_SIZE, random_state=42)
            
            # Train models
            self.train_yield_model(X_train, y_train, X_val, y_val)
            self.train_recommendation_model(X_original, X_scaled)
            
            # Stage 3: Saving and Testing
            self.log("="*50 + "\nSTAGE 3: SAVING & TESTING\n" + "="*50)
            self.save_models()
            self.test_models()

            duration = time.time() - start_time
            self.log(f"TRAINING COMPLETED SUCCESSFULLY! (Duration: {duration:.2f}s)")
            self.log("="*50)
            self.log(f"Models saved in: {self.config.MODELS_DIR}")
            self.log("Ready for backend deployment!")
            
        except Exception as e:
            self.log(f"TRAINING FAILED: {e}")
            raise

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Crop Yield Prediction Models')
    parser.add_argument('--data', type=str, help='Path to training data CSV file')
    parser.add_argument('--epochs-yield', type=int, default=100, help='Epochs for yield model')
    parser.add_argument('--epochs-rec', type=int, default=80, help='Epochs for recommendation model')
    parser.add_argument('--samples', type=int, default=15000, help='Number of synthetic samples')
    parser.add_argument('--production', action='store_true', help='Use production settings')
    
    args = parser.parse_args()
    
    # Configure training
    config = Config()
    if args.production:
        config.YIELD_EPOCHS = 150
        config.REC_EPOCHS = 120
        config.SYNTHETIC_SAMPLES = 25000
        print("Using production settings")
    
    if args.epochs_yield:
        config.YIELD_EPOCHS = args.epochs_yield
    if args.epochs_rec:
        config.REC_EPOCHS = args.epochs_rec
    if args.samples:
        config.SYNTHETIC_SAMPLES = args.samples
    
    # Run training
    trainer = ModelTrainer(config)
    trainer.train_complete_pipeline(args.data)

if __name__ == "__main__":
    main()