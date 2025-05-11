import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, 'vietnam_housing_dataset.csv')
MODEL_PATH = os.path.join(CURRENT_DIR, 'house_price_model.joblib')
SCALER_PATH = os.path.join(CURRENT_DIR, 'scaler.joblib')
FEATURES_PATH = os.path.join(CURRENT_DIR, 'selected_features.joblib')


def prepare_data():
    logger.info("=== BẮT ĐẦU CHUẨN BỊ DỮ LIỆU ===")
    df = pd.read_csv(DATASET_PATH, encoding='cp1252')
    # Bỏ các cột không cần thiết
    df.drop(columns=['Address'], errors='ignore', inplace=True)

    numeric_features = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
    categorical_features = ['House direction', 'Balcony direction', 'Legal status', 'Furniture state']

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
        logger.info(f"[NUMERIC] {col} - min: {df[col].min()}, max: {df[col].max()}, mean: {df[col].mean():.2f}")

    legal_status_mapping = {
        'Sổ đỏ': 'Have certificate',
        'Sổ hồng': 'Have certificate',
        'Giấy tờ hợp lệ': 'Have certificate',
        'Đang chờ sổ': 'Sale contract',
        'Khác': 'Sale contract'
    }
    df['Legal status'] = df['Legal status'].map(legal_status_mapping).fillna('null')

    furniture_mapping = {
        'Không nội thất': 'none',
        'Nội thất cơ bản': 'basic',
        'Đầy đủ nội thất': 'full',
        'Cao cấp': 'full'
    }
    df['Furniture state'] = df['Furniture state'].map(furniture_mapping).fillna('null')
    # Điền giá trị thiếu cho các cột phân loại
    for col in categorical_features:
        df[col] = df[col].fillna('null')
        logger.info(f"[CATEGORICAL] {col} unique values: {df[col].unique()}")

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=categorical_features, drop_first=False)
    print("Số cột sau one-hot encoding:", len(df.columns))
    # Feature Engineering
    df['Area_per_floor'] = df['Area'] / df['Floors'].clip(lower=1)
    df['Area_per_room'] = df['Area'] / (df['Bedrooms'] + df['Bathrooms']).clip(lower=1)
    df['Frontage_ratio'] = df['Frontage'] / np.sqrt(df['Area'].clip(lower=1))
    df['Area_squared'] = df['Area'] ** 2
    df['Area_log'] = np.log1p(df['Area'])

    features = [col for col in df.columns if col != 'Price']
    X = df[features]
    y = df['Price']

    return X, y, features, df


def train_model(X, y, features):
    logger.info("\n=== HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    #Linear regression
    linear = LinearRegression().fit(X_train_scaled, y_train)
    #Ridge regression
    ridge = RidgeCV(alphas=np.logspace(-3, 2, 50), cv=5).fit(X_train_scaled, y_train)
    logger.info(f"Ridge alpha selected: {ridge.alpha_:.6f}")
    #Lasso regression
    lasso = LassoCV(alphas=np.logspace(-3, 2, 50), cv=5, max_iter=10000).fit(X_train_scaled, y_train)
    logger.info(f"Lasso alpha selected: {lasso.alpha_:.6f}")
    
    models = {
        'Linear': linear,
        'Ridge': ridge,
        'Lasso': lasso
    }
    
    logger.info("\n=== KẾT QUẢ ĐÁNH GIÁ CÁC MÔ HÌNH ===")
    
    for name, model in models.items():
        train_r2 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2').mean()
        test_pred = model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        logger.info(f"{name} - R² Score (train): {train_r2:.4f}")
        logger.info(f"{name} - R² Score (test): {test_r2:.4f}")
        logger.info(f"{name} - RMSE (test): {test_rmse:.2f} ")
    
    # chọn model tốt nhất dựa trên R² Score
    best_model_name = max(models.keys(), key=lambda name: r2_score(y_test, models[name].predict(X_test_scaled)))
    best_model = models[best_model_name]
    logger.info(f"\nBest model selected: {best_model_name}")
    
    y_pred = best_model.predict(X_test_scaled)
    
    analyze_coefficients(best_model, features, X_train, y_train, best_model_name)
    plot_predictions(y_test, y_pred)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(features, FEATURES_PATH)
    logger.info("Đã lưu mô hình và thông tin thành công")
    return best_model, scaler


def analyze_coefficients(model, features, X, y, model_name):
    """Phân tích hệ số hồi quy và hiển thị phương trình"""
    logger.info("\n=== PHÂN TÍCH HỆ SỐ HỒI QUY ===")
    
    if hasattr(model, 'coef_') and len(model.coef_) == len(features):
        try:
            importance = pd.DataFrame({
                'Feature': features,
                'Coefficient': model.coef_,
                'Abs_Coefficient': abs(model.coef_)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            logger.info(f"\nPhương trình hồi quy tuyến tính ({model_name}):")
            
            significant_coefs = [(f"({coef:.4f})×{feat}", coef) for feat, coef in zip(features, model.coef_) if abs(coef) > 1e-4]
            equation_terms = [term for term, _ in significant_coefs]
            
            if hasattr(model, 'intercept_'):
                intercept = model.intercept_
                equation = "Price = " + " + ".join(equation_terms) + f" + {intercept:.4f}"
            else:
                equation = "Price = " + " + ".join(equation_terms)
                
            logger.info(equation)
            
            logger.info("\nTop 10 features ảnh hưởng lớn nhất:")
            for idx, row in importance.head(10).iterrows():
                logger.info(f"{row['Feature']}: {row['Coefficient']:+.4f}")
            
            # Vẽ biểu đồ
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Abs_Coefficient', y='Feature', data=importance.head(10))
            plt.title(f"Top 10 Features Ảnh Hưởng Tới Giá Nhà ({model_name})")
            plt.tight_layout()
            plt.savefig(os.path.join(CURRENT_DIR, f'top_10_features_{model_name.lower()}.png'))
            
        except Exception as e:
            logger.error(f"Error analyzing coefficients: {str(e)}")
    else:
        logger.warning("Model does not have coefficients in the expected format")


def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Giá thực tế (tỷ đồng)')
    plt.ylabel('Giá dự đoán (tỷ đồng)')
    plt.title('So sánh giá thực tế và dự đoán')
    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, 'prediction_vs_actual.png'))
    
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Giá dự đoán (tỷ đồng)')
    plt.ylabel('Residuals (tỷ đồng)')
    plt.title('Biểu đồ Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, 'residuals.png'))


def main():
    X, y, features, df = prepare_data()
    
    data_with_price = X.copy()
    data_with_price['Price'] = y
    
    correlation_matrix = data_with_price.corr()
    
    logger.info("\nTop 10 features có tương quan mạnh nhất với giá:")
    correlations_with_price = correlation_matrix['Price'].abs()
    correlations_with_price = correlations_with_price[correlations_with_price.index != 'Price']
    logger.info(correlations_with_price.sort_values(ascending=False).head(10))
    
    train_model(X, y, features)


if __name__ == '__main__':
    main()