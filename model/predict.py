import joblib
import numpy as np
import pandas as pd
import os
import logging
import sys
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

# Cập nhật các giá trị Legal Status và Furniture State theo yêu cầu
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
    """Load model và trích xuất thông tin về phương trình hồi quy và hàm chi phí"""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        
        # Xác định loại model
        model_type = type(model).__name__
        
        # Tạo phương trình hồi quy
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            equation_terms = []
            for feat, coef in zip(features, model.coef_):
                if abs(coef) > 1e-4:  # Chỉ hiển thị các hệ số có giá trị đáng kể
                    equation_terms.append(f"({coef:.4f})×{feat}")
            
            equation = "Price = " + " + ".join(equation_terms) + f" + {model.intercept_:.4f}"
            
            # Xác định hàm chi phí tương ứng với từng loại model
            if "LinearRegression" in model_type:
                cost_function = "J(θ) = (1/2m) * Σ(h_θ(x^(i)) - y^(i))²"
            elif "Ridge" in model_type:
                alpha = model.alpha_ if hasattr(model, 'alpha_') else "alpha"
                cost_function = f"J(θ) = (1/2m) * Σ(h_θ(x^(i)) - y^(i))² + {alpha} * Σθ_j²"
            elif "Lasso" in model_type:
                alpha = model.alpha_ if hasattr(model, 'alpha_') else "alpha"
                cost_function = f"J(θ) = (1/2m) * Σ(h_θ(x^(i)) - y^(i))² + {alpha} * Σ|θ_j|"
            else:
                cost_function = "Không có thông tin hàm chi phí cho mô hình này"
        else:
            equation = "Không có thông tin phương trình cho mô hình này"
            cost_function = "Không có thông tin hàm chi phí cho mô hình này"
        
        return {
            'model': model,
            'scaler': scaler,
            'features': features,
            'model_type': model_type,
            'equation': equation,
            'cost_function': cost_function
        }
    except Exception as e:
        logger.error(f"Lỗi khi tải thông tin mô hình: {str(e)}")
        raise

def prepare_input_features(input_data):
    """Chuẩn bị features từ input data theo cùng format với lúc training"""
    try:
        # Load thông tin mô hình và danh sách features
        model_info = load_model_info()
        features = model_info['features']
        
        # Parse các giá trị numeric cơ bản
        basic_numeric_cols = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
        df = pd.DataFrame([input_data[:len(basic_numeric_cols)]], columns=basic_numeric_cols)
        
        # Phân tích dữ liệu categorical từ input
        start_idx = len(basic_numeric_cols)
        
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
        
        # Feature Engineering - tạo các features phái sinh
        df['Area_per_floor'] = df['Area'] / df['Floors'].clip(lower=1)
        df['Area_per_room'] = df['Area'] / (df['Bedrooms'] + df['Bathrooms']).clip(lower=1)
        df['Frontage_ratio'] = df['Frontage'] / np.sqrt(df['Area'].clip(lower=1))
        
        # Đảm bảo tất cả features cần thiết có mặt trong DataFrame
        missing_features = set(features) - set(df.columns)
        for feature in missing_features:
            df[feature] = 0.0
            
        # Sắp xếp các cột theo đúng thứ tự training
        df = df[features]
        
        return df
        
    except Exception as e:
        logger.error(f"Lỗi khi chuẩn bị features: {str(e)}")
        raise

def predict_price(input_data):
    """
    Dự đoán giá nhà dựa trên input features
    
    Args:
        input_data: List các features theo thứ tự
        
    Returns:
        dict: Kết quả dự đoán và thông tin mô hình
    """
    try:
        # Load thông tin mô hình
        model_info = load_model_info()
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Chuẩn bị features
        df = prepare_input_features(input_data)
        
        # Chuẩn hóa dữ liệu
        X_scaled = scaler.transform(df)
        
        # Dự đoán
        prediction = model.predict(X_scaled)[0]
        
        # Tạo kết quả trả về với thông tin bổ sung
        result = {
            'predicted_price': float(prediction),
            'model_type': model_info['model_type'],
            'equation': model_info['equation'],
            'cost_function': model_info['cost_function']
        }
        
        logger.info(f"Kết quả dự đoán: {prediction:.2f} tỷ đồng")
        logger.info(f"Loại mô hình: {model_info['model_type']}")
        logger.info(f"Phương trình hồi quy: {model_info['equation']}")
        logger.info(f"Hàm chi phí: {model_info['cost_function']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
        raise

def display_feature_details():
    """Hiển thị chi tiết về các features và giá trị của chúng"""
    logger.info("=== THÔNG TIN CHI TIẾT VỀ CÁC FEATURES ===")
    
    # Numeric features
    logger.info("\nNumeric features:")
    logger.info("- Area: Diện tích nhà (m²)")
    logger.info("- Frontage: Mặt tiền (m)")
    logger.info("- Access Road: Độ rộng đường trước nhà (m)")
    logger.info("- Floors: Số tầng")
    logger.info("- Bedrooms: Số phòng ngủ")
    logger.info("- Bathrooms: Số phòng tắm")
    
    # Categorical features
    logger.info("\nHouse direction (Hướng nhà):")
    for col in HOUSE_DIRECTION_COLS:
        logger.info(f"- {col}")
    
    logger.info("\nBalcony direction (Hướng ban công):")
    for col in BALCONY_DIRECTION_COLS:
        logger.info(f"- {col}")
    
    logger.info("\nLegal status (Tình trạng pháp lý):")
    for col in LEGAL_STATUS_COLS:
        logger.info(f"- {col}")
    
    logger.info("\nFurniture state (Tình trạng nội thất):")
    for col in FURNITURE_STATE_COLS:
        logger.info(f"- {col}")
    
    # Engineered features
    logger.info("\nEngineered features:")
    logger.info("- Area_per_floor: Diện tích trung bình mỗi tầng")
    logger.info("- Area_per_room: Diện tích trung bình mỗi phòng")
    logger.info("- Frontage_ratio: Tỷ lệ mặt tiền so với diện tích")

def main():
    try:
        if len(sys.argv) < 2 or sys.argv[1] == "--help":
            logger.info("Sử dụng: python predict.py [--info] hoặc python predict.py [các giá trị features]")
            logger.info("  --info: Hiển thị thông tin về mô hình và features")
            logger.info("  Nếu cung cấp giá trị features, thứ tự phải là:")
            logger.info("  Area, Frontage, Access Road, Floors, Bedrooms, Bathrooms, [one-hot features cho các categorical]")
            if len(sys.argv) >= 2 and sys.argv[1] == "--help":
                display_feature_details()
            sys.exit(0)
            
        if sys.argv[1] == "--info":
            # Hiển thị thông tin mô hình
            model_info = load_model_info()
            logger.info(f"Loại mô hình: {model_info['model_type']}")
            logger.info(f"Phương trình hồi quy: {model_info['equation']}")
            logger.info(f"Hàm chi phí: {model_info['cost_function']}")
            display_feature_details()
            sys.exit(0)
        
        # Chuyển đổi input thành số
        input_data = []
        for arg in sys.argv[1:]:
            try:
                value = float(arg)
                input_data.append(value)
            except ValueError:
                logger.error(f"Giá trị không hợp lệ: {arg}")
                sys.exit(1)
        
        # Dự đoán giá
        result = predict_price(input_data)
        
        # In kết quả
        print(json.dumps(result, ensure_ascii=False))
        logger.info(f"Giá dự đoán: {result['predicted_price']:.2f} tỷ đồng")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()