import joblib
import numpy as np
import pandas as pd
import os
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HOUSE_DIRECTION_COLS = [
    'House direction_null', 
    'House direction_Bắc', 
    'House direction_Nam',
    'House direction_Đông',
    'House direction_Tây',
    'House direction_Đông - Bắc',
    'House direction_Tây - Bắc', 
    'House direction_Đông - Nam',
    'House direction_Tây - Nam'
]

BALCONY_DIRECTION_COLS = [
    'Balcony direction_null',
    'Balcony direction_Bắc',
    'Balcony direction_Nam', 
    'Balcony direction_Đông',
    'Balcony direction_Tây',
    'Balcony direction_Đông - Bắc',
    'Balcony direction_Tây - Bắc', 
    'Balcony direction_Đông - Nam',
    'Balcony direction_Tây - Nam'
]

LEGAL_STATUS_COLS = [
    'Legal status_null',
    'Legal status_Have certificate',
    'Legal status_Sale contract'
]

FURNITURE_STATE_COLS = [
    'Furniture state_null',
    'Furniture state_basic',
    'Furniture state_full'
]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'house_price_model.joblib')
SCALER_PATH = os.path.join(CURRENT_DIR, 'scaler.joblib')
FEATURES_PATH = os.path.join(CURRENT_DIR, 'selected_features.joblib')

def load_model_info():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        
        return {
            'model': model,
            'scaler': scaler,
            'features': features
        }
    except Exception as e:
        raise

def prepare_input_features(input_data):
    try:
        model_info = load_model_info()
        features = model_info['features']
        
        numeric_data = {
            'Area': float(input_data[0]),
            'Frontage': float(input_data[1]),
            'Access Road': float(input_data[2]),
            'Floors': int(input_data[3]),
            'Bedrooms': int(input_data[4]),
            'Bathrooms': int(input_data[5])
        }
        
        # Tạo DataFrame với numeric features
        df = pd.DataFrame([numeric_data])
        
        # Xử lý categorical features với one-hot encoding
        start_idx = 6
        
        # House direction (9 values)
        for i, col in enumerate(HOUSE_DIRECTION_COLS):
            if start_idx + i < len(input_data):
                df[col] = float(input_data[start_idx + i])
            else:
                df[col] = 0.0
        start_idx += len(HOUSE_DIRECTION_COLS)
        
        # Balcony direction (9 values)
        for i, col in enumerate(BALCONY_DIRECTION_COLS):
            if start_idx + i < len(input_data):
                df[col] = float(input_data[start_idx + i])
            else:
                df[col] = 0.0
        start_idx += len(BALCONY_DIRECTION_COLS)
        
        # Legal status (3 values)
        for i, col in enumerate(LEGAL_STATUS_COLS):
            if start_idx + i < len(input_data):
                df[col] = float(input_data[start_idx + i])
            else:
                df[col] = 0.0
        start_idx += len(LEGAL_STATUS_COLS)
        
        # Furniture state (3 values)
        for i, col in enumerate(FURNITURE_STATE_COLS):
            if start_idx + i < len(input_data):
                df[col] = float(input_data[start_idx + i])
            else:
                df[col] = 0.0
        
        # Feature Engineering
        df['Area_per_floor'] = df['Area'] / df['Floors'].clip(lower=1)
        df['Area_per_room'] = df['Area'] / (df['Bedrooms'] + df['Bathrooms']).clip(lower=1)
        df['Frontage_ratio'] = df['Frontage'] / np.sqrt(df['Area'].clip(lower=1))
        df['Area_squared'] = df['Area'] ** 2
        df['Area_log'] = np.log1p(df['Area'])
        
        missing_features = set(features) - set(df.columns)
        if missing_features:
            for feature in missing_features:
                df[feature] = 0
        
        df = df[features]
        
        return df
        
    except Exception as e:
        raise

def predict_price(input_data):
    try:
        model_info = load_model_info()
        model = model_info['model']
        scaler = model_info['scaler']
        df = prepare_input_features(input_data)
        X_scaled = scaler.transform(df)
        prediction = model.predict(X_scaled)[0]
        
        return float(prediction)
        
    except Exception as e:
        raise