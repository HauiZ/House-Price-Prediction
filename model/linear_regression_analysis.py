import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import logging
import sys

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lấy đường dẫn tuyệt đối của thư mục hiện tại
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, 'vietnam_housing_dataset.csv')
MODEL_PATH = os.path.join(CURRENT_DIR, 'linear_regression_model.joblib')
SCALER_PATH = os.path.join(CURRENT_DIR, 'linear_regression_scaler.joblib')
FEATURES_PATH = os.path.join(CURRENT_DIR, 'linear_regression_features.joblib')

def prepare_data():
    """Chuẩn bị dữ liệu cho mô hình hồi quy tuyến tính"""
    logger.info("=== CHUẨN BỊ DỮ LIỆU ===")
    
    # Đọc dữ liệu
    encodings = ['utf-8', 'latin1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(DATASET_PATH, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError("Không thể đọc file dataset")
    
    # Định nghĩa features
    numeric_features = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
    categorical_features = ['House direction', 'Balcony direction', 'Legal status', 'Furniture state']
    
    # Xử lý missing values cho numeric features
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median = df[col].median()
        df[col] = df[col].fillna(median)
        logger.info(f"Điền missing values cho {col} với giá trị trung vị: {median:.2f}")
    
    # Xử lý categorical features
    for col in categorical_features:
        df[col] = df[col].fillna('Không xác định')
    
    # One-hot encoding cho categorical features
    df_encoded = pd.get_dummies(df[categorical_features])
    
    # Feature Engineering thông minh
    logger.info("\nThêm các features phái sinh:")
    
    # 1. Tỷ lệ và mối quan hệ giữa các features số học
    df['Area_per_floor'] = df['Area'] / df['Floors'].clip(lower=1)
    logger.info("- Area_per_floor: Diện tích trung bình mỗi tầng")
    
    df['Room_density'] = (df['Bedrooms'] + df['Bathrooms']) / df['Area']
    logger.info("- Room_density: Mật độ phòng trên diện tích")
    
    df['Frontage_ratio'] = df['Frontage'] / np.sqrt(df['Area'])
    logger.info("- Frontage_ratio: Tỷ lệ mặt tiền trên căn bậc hai diện tích")
    
    # 2. Biến đổi phi tuyến cho các features quan trọng
    df['Area_squared'] = df['Area'] ** 2
    logger.info("- Area_squared: Diện tích bình phương (phi tuyến)")
    
    df['Area_log'] = np.log1p(df['Area'])
    logger.info("- Area_log: Log của diện tích (phi tuyến)")
    
    # Kết hợp tất cả features
    engineered_features = ['Area_per_floor', 'Room_density', 'Frontage_ratio', 
                          'Area_squared', 'Area_log']
    
    X = pd.concat([
        df[numeric_features],
        df_encoded,
        df[engineered_features]
    ], axis=1)
    
    y = df['Price']
    
    # Lưu tên các features
    features = X.columns.tolist()
    
    return X, y, features

def analyze_features(X, y, features):
    """Phân tích mối quan hệ giữa các features và biến mục tiêu"""
    logger.info("\n=== PHÂN TÍCH MỐI QUAN HỆ GIỮA CÁC FEATURES ===")
    
    # Tính ma trận tương quan
    data = X.copy()
    data['Price'] = y
    correlation_matrix = data.corr()
    
    # Vẽ heatmap tương quan
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Ma trận tương quan giữa các features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # In top 10 features có tương quan mạnh nhất với giá
    price_correlations = correlation_matrix['Price'].sort_values(ascending=False)
    logger.info("\nTop 10 features có tương quan mạnh nhất với giá:")
    for feature, corr in price_correlations[1:11].items():
        logger.info(f"{feature:30} {corr:+.4f}")
    
    return correlation_matrix

def train_and_evaluate_model(X, y, features):
    """Huấn luyện và đánh giá mô hình hồi quy tuyến tính"""
    logger.info("\n=== HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH ===")
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Đánh giá trên tập test
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Tính các metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    # In kết quả đánh giá
    logger.info("\nKết quả đánh giá mô hình:")
    logger.info(f"R² Score (training): {train_r2:.4f}")
    logger.info(f"R² Score (test): {test_r2:.4f}")
    logger.info(f"RMSE (training): {train_rmse:.2f} tỷ đồng")
    logger.info(f"RMSE (test): {test_rmse:.2f} tỷ đồng")
    logger.info(f"Cross-validation R² scores: {cv_scores}")
    logger.info(f"Mean CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Phân tích hệ số hồi quy
    analyze_coefficients(model, features, X_train_scaled, y_train)
    
    return model, scaler

def analyze_coefficients(model, features, X_scaled, y):
    """Phân tích và giải thích các hệ số hồi quy"""
    logger.info("\n=== PHÂN TÍCH HỆ SỐ HỒI QUY ===")
    
    # Tính khoảng tin cậy cho các hệ số
    n = X_scaled.shape[0]  # Số lượng mẫu
    p = X_scaled.shape[1]  # Số lượng features
    dof = n - p - 1       # Bậc tự do
    
    # Tính MSE
    y_pred = model.predict(X_scaled)
    mse = np.sum((y - y_pred) ** 2) / dof
    
    # Tính ma trận hiệp phương sai
    var_coef = mse * np.linalg.inv(X_scaled.T @ X_scaled).diagonal()
    std_coef = np.sqrt(var_coef)
    
    # Tính khoảng tin cậy 95%
    t_value = 1.96  # Cho khoảng tin cậy 95%
    ci_lower = model.coef_ - t_value * std_coef
    ci_upper = model.coef_ + t_value * std_coef
    
    # In phương trình hồi quy và phân tích
    logger.info("\nPhương trình hồi quy tuyến tính:")
    equation = "Giá = "
    terms = []
    
    # Tính toán và lưu các hệ số
    coefficients = []
    for i, (feature, coef) in enumerate(zip(features, model.coef_)):
        coefficients.append((feature, coef))
        
        # Thêm vào phương trình nếu hệ số đáng kể
        if abs(coef) > 1e-4:
            term = f"{coef:+.4f}×{feature}"
            terms.append(term)
    
    # Ghép phương trình hoàn chỉnh
    equation += " ".join(terms) + f" + {model.intercept_:.4f}"
    logger.info(equation)
    
    # In top 10 features quan trọng nhất
    logger.info("\nTop 10 features ảnh hưởng lớn nhất đến giá:")
    importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_,
        'Abs_Coefficient': abs(model.coef_),
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
    })
    importance = importance.sort_values('Abs_Coefficient', ascending=False)
    
    for idx, row in importance.head(10).iterrows():
        logger.info(f"\n{row['Feature']}:")
        logger.info(f"  - Hệ số: {row['Coefficient']:+.4f}")
        logger.info(f"  - Khoảng tin cậy 95%: [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]")
        logger.info(f"  - % Ảnh hưởng: {row['Abs_Coefficient']/importance['Abs_Coefficient'].sum()*100:.2f}%")

def main():
    try:
        logger.info("=== BẮT ĐẦU PHÂN TÍCH HỒI QUY TUYẾN TÍNH ===")
        
        # Chuẩn bị dữ liệu
        X, y, features = prepare_data()
        
        # Phân tích features
        correlation_matrix = analyze_features(X, y, features)
        
        # Huấn luyện và đánh giá mô hình
        model, scaler = train_and_evaluate_model(X, y, features)
        
        # Lưu mô hình và thông tin
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(features, FEATURES_PATH)
        logger.info("\nĐã lưu mô hình và thông tin thành công")
        
        logger.info("\n=== HOÀN THÀNH PHÂN TÍCH HỒI QUY TUYẾN TÍNH ===")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình phân tích: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Lỗi chương trình: {str(e)}")
        sys.exit(1) 