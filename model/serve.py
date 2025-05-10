import sys
import joblib
import numpy as np
import pandas as pd
import os
import logging
import json

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Định nghĩa mapping cho các giá trị categorical và tên cột
HOUSE_DIRECTION_COLS = [
    'House direction_null',
    'House direction_Bắc', 
    'House direction_Nam',
    'House direction_Đông',
    'House direction_Tây',
    'House direction_Đông Bắc',
    'House direction_Tây Bắc', 
    'House direction_Đông Nam',
    'House direction_Tây Nam'
]

BALCONY_DIRECTION_COLS = [
    'Balcony direction_null',
    'Balcony direction_Bắc',
    'Balcony direction_Nam', 
    'Balcony direction_Đông',
    'Balcony direction_Tây',
    'Balcony direction_Đông Bắc',
    'Balcony direction_Tây Bắc', 
    'Balcony direction_Đông Nam',
    'Balcony direction_Tây Nam'
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

# Lấy đường dẫn tuyệt đối của thư mục hiện tại
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'house_price_model.joblib')
SCALER_PATH = os.path.join(CURRENT_DIR, 'scaler.joblib')
FEATURES_PATH = os.path.join(CURRENT_DIR, 'selected_features.joblib')

def load_model_info():
    """Load model và các thông tin cần thiết"""
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
        logger.error(f"Error loading model information: {str(e)}")
        raise

def prepare_input_features(input_data):
    """Chuẩn bị features từ input data theo cùng format với lúc training"""
    try:
        model_info = load_model_info()
        features = model_info['features']
        
        # Parse input data - chuyển từ array sang dict
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
            df[col] = float(input_data[start_idx + i])
        start_idx += len(HOUSE_DIRECTION_COLS)
        
        # Balcony direction (9 values)
        for i, col in enumerate(BALCONY_DIRECTION_COLS):
            df[col] = float(input_data[start_idx + i])
        start_idx += len(BALCONY_DIRECTION_COLS)
        
        # Legal status (3 values)
        for i, col in enumerate(LEGAL_STATUS_COLS):
            df[col] = float(input_data[start_idx + i])
        start_idx += len(LEGAL_STATUS_COLS)
        
        # Furniture state (3 values)
        for i, col in enumerate(FURNITURE_STATE_COLS):
            df[col] = float(input_data[start_idx + i])
        
        # Feature Engineering
        df['Area_per_floor'] = df['Area'] / df['Floors'].clip(lower=1)
        df['Area_per_room'] = df['Area'] / (df['Bedrooms'] + df['Bathrooms']).clip(lower=1)
        df['Frontage_ratio'] = df['Frontage'] / np.sqrt(df['Area'].clip(lower=1))
        
        # Đảm bảo tất cả các cột cần thiết đều có mặt
        missing_features = set(features) - set(df.columns)
        if missing_features:
            for feature in missing_features:
                df[feature] = 0
        
        # Sắp xếp các cột theo đúng thứ tự training
        df = df[features]
        
        return df
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise

def predict_price(input_data):
    """Dự đoán giá nhà dựa trên input features"""
    try:
        # Load model và thông tin
        model_info = load_model_info()
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Chuẩn bị features
        df = prepare_input_features(input_data)
        
        # Chuẩn hóa dữ liệu
        X_scaled = scaler.transform(df)
        
        # Dự đoán
        prediction = model.predict(X_scaled)[0]
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error predicting: {str(e)}")
        raise

def main():
    try:
        # Đọc input args từ command line
        input_data = []
        for arg in sys.argv[1:]:
            try:
                value = float(arg)
                input_data.append(value)
            except ValueError:
                logger.error(f"Invalid input value: {arg}")
                sys.exit(1)
        
        # Kiểm tra đầu vào
        if len(input_data) < 6:
            logger.error("Not enough input parameters")
            sys.exit(1)
            
        # Dự đoán giá
        prediction = predict_price(input_data)
        
        # In kết quả (chỉ giá trị dự đoán) để Node.js có thể đọc
        print(prediction)
        
    except Exception as e:
        logger.error(f"Error in prediction service: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()