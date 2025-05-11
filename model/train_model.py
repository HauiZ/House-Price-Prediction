import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
print("===== Bước 1: Đọc dữ liệu =====")

df = pd.read_csv('./vietnam_housing_dataset.csv', encoding='cp1252')

print(f"Đã đọc {df.shape[0]} dòng dữ liệu với {df.shape[1]} cột")

# In thông tin cơ bản về dữ liệu
print("\n===== Bước 2: Kiểm tra dữ liệu =====")
print("Thông tin các cột:")
print(df.columns.tolist())
print("\nGiá trị null trong dữ liệu:")
print(df.isnull().sum())

# Xử lý ngoại lai (outliers) bằng phương pháp IQR
print("\n===== Bước 3: Xử lý dữ liệu ngoại lai (outliers) =====")
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # Tứ phân vị thứ nhất (25%)
    Q3 = df[column].quantile(0.75)  # Tứ phân vị thứ ba (75%)
    IQR = Q3 - Q1                   # Khoảng tứ phân vị
    lower_bound = Q1 - 1.5 * IQR    # Ngưỡng dưới = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5 * IQR    # Ngưỡng trên = Q3 + 1.5*IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

numeric_columns = ['Price', 'Area', 'Frontage', 'Access Road']
original_rows = df.shape[0]
for col in numeric_columns:
    df = remove_outliers(df, col)
    print(f"Xử lý outliers cho cột {col}: {original_rows - df.shape[0]} dòng bị loại bỏ")
    original_rows = df.shape[0]

print("\n===== Bước 4: Xử lý giá trị thiếu =====")
df['House direction'] = df['House direction'].fillna('Không xác định')
df['Balcony direction'] = df['Balcony direction'].fillna('Không xác định')
df['Legal status'] = df['Legal status'].fillna('Khác')
df['Furniture state'] = df['Furniture state'].fillna('Không nội thất')



# Xử lý missing values cho cột numeric bằng giá trị trung bình
numeric_cols_with_missing = ['Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
for col in numeric_cols_with_missing:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"Điền giá trị thiếu cho cột {col} bằng giá trị ở giữa: {median_value:.2f}")

print("\n===== Bước 5: Tạo biến phụ thuộc (Y) =====") 
print(f"Đã chuyển đổi biến Price sang Log_Price để giảm độ lệch")
#add feature
df['Area_Bathrooms'] = df['Area'] * df['Bathrooms']
df['Area_Floors'] = df['Area'] * df['Floors']
df['Frontage_AccessRoad'] = df['Frontage'] * df['Access Road']
df['Total_Rooms'] = df['Bedrooms'] + df['Bathrooms']

# Add non-linear features
df['Log_Area'] = np.log1p(df['Area'])
df['Log_Price'] = np.log1p(df['Price'])

# Định nghĩa các đặc trưng (features)
print("\n===== Bước 6: Định nghĩa đặc trưng cho mô hình =====")
numeric_cols = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms',
                'Area_Bathrooms', 'Area_Floors', 'Frontage_AccessRoad', 
                 'Log_Area', 'Total_Rooms']
print(f"Đặc trưng số học cơ bản: {numeric_cols}")
categorical_cols = ['House direction', 'Balcony direction', 'Legal status', 'Furniture state']
print(f"Đặc trưng phân loại: {categorical_cols}")

# Chuẩn bị dữ liệu cho mô hình
X = df[numeric_cols + categorical_cols].copy()
y = df['Log_Price'].values

print("\n===== Bước 7: Chia dữ liệu train/test =====")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Tập huấn luyện: {X_train.shape[0]} mẫu")
print(f"Tập kiểm tra: {X_test.shape[0]} mẫu")

# Tạo pipeline xử lý đặc trưng
print("\n===== Bước 8: Tạo pipeline tiền xử lý =====")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        # Mã hóa one-hot cho các đặc trưng phân loại
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# Tạo và huấn luyện mô hình Random Forest
print("\n===== Bước 9: Tạo và huấn luyện mô hình Random Forest =====")
random_forest = RandomForestRegressor(
    n_estimators=300,       # Số lượng cây trong rừng 
    max_depth=None,         # Để None cho phép cây phát triển đầy đủ
    min_samples_split=2,    # Số mẫu tối thiểu cần thiết để phân tách một nút
    min_samples_leaf=1,     # Số mẫu tối thiểu để tạo thành một lá
    random_state=42
)

# Tạo pipeline hoàn chỉnh
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', random_forest)
])

# Huấn luyện mô hình
print("Đang huấn luyện mô hình Random Forest...")
pipeline.fit(X_train, y_train)
print("Huấn luyện hoàn tất!")

# Đánh giá mô hình
print("\n===== Bước 10: Đánh giá mô hình =====")
# Dự đoán trên tập huấn luyện và tập kiểm tra (giá trị log)
train_pred_log = pipeline.predict(X_train)
test_pred_log = pipeline.predict(X_test)

# Chuyển đổi giá trị dự đoán về thang đo gốc (VND)
train_pred_original = np.expm1(train_pred_log)  # expm1 = exp(x) - 1
test_pred_original = np.expm1(test_pred_log)
y_train_original = np.expm1(y_train)
y_test_original = np.expm1(y_test)

# Tính các chỉ số đánh giá ở thang đo gốc
train_r2 = r2_score(y_train_original, train_pred_original)
test_r2 = r2_score(y_test_original, test_pred_original)
test_rmse = np.sqrt(mean_squared_error(y_test_original, test_pred_original))

print(f"R² trên tập huấn luyện: {train_r2:.4f}")
print(f"R² trên tập kiểm tra: {test_r2:.4f}")
print(f"RMSE trên tập kiểm tra: {test_rmse:.4f}")

# Thêm phần vẽ đồ thị so sánh giá trị thực tế và dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, test_pred_original, alpha=0.5)
plt.plot([min(y_test_original), max(y_test_original)], 
         [min(y_test_original), max(y_test_original)], 'r--')
plt.xlabel('Giá trị thực tế (VND)')
plt.ylabel('Giá trị dự đoán (VND)')
plt.title('So sánh giá thực tế và dự đoán')
plt.show()

# Tạo thêm biểu đồ tập trung vào phân phối sai số
plt.figure(figsize=(10, 6))
errors = test_pred_original - y_test_original
plt.hist(errors, bins=50)
plt.xlabel('Sai số dự đoán (VND)')
plt.ylabel('Số lượng')
plt.title('Phân phối sai số dự đoán')
plt.show()

# Hiển thị các đặc trưng quan trọng
print("\n===== Bước 11: Phân tích tầm quan trọng của đặc trưng =====")
# Lấy tên của tất cả các đặc trưng sau khi đã qua tiền xử lý
feature_names = (
    numeric_cols + 
    [f"{col}_{val}" for col, vals in 
     zip(categorical_cols, pipeline['preprocessor'].named_transformers_['cat'].categories_) 
     for val in vals]
) 

# Lấy điểm quan trọng của các đặc trưng
importances = pipeline['regressor'].feature_importances_
importance_dict = dict(zip(feature_names, importances))
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

print("Top 10 đặc trưng quan trọng nhất:")
for name, imp in sorted_importance[:10]:
    print(f"{name}: {imp:.4f} ({imp*100:.2f}%)")

# Vẽ biểu đồ top 10 đặc trưng quan trọng nhất
plt.figure(figsize=(12, 6))
top_features = dict(sorted_importance[:10])
plt.barh(list(top_features.keys()), list(top_features.values()))
plt.xlabel('Mức độ quan trọng')
plt.title('Top 10 đặc trưng quan trọng nhất')
plt.gca().invert_yaxis()  # Để đặc trưng quan trọng nhất ở trên cùng
plt.tight_layout()
plt.show()

# Lưu mô hình đã huấn luyện
print("\n===== Bước 12: Lưu mô hình =====")
joblib.dump(pipeline, 'house_price_model.pkl')

# Lưu thông tin đặc trưng để sử dụng khi dự đoán
feature_info = {
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'categorical_values': {col: sorted(df[col].unique().tolist()) for col in categorical_cols}
}
joblib.dump(feature_info, 'feature_info.pkl')
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sắp xếp giảm dần
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.barh(feature_df['Feature'], feature_df['Importance'])
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance')
plt.title('Feature Importance từ RandomForest')
plt.tight_layout()
plt.show()
print("Đã lưu mô hình Random Forest và thông tin đặc trưng thành công!")
print("\nMô hình này có thể được sử dụng để dự đoán giá nhà dựa trên các đặc điểm đầu vào.")
print("Sử dụng predict.py để thực hiện dự đoán với mô hình đã huấn luyện.")
