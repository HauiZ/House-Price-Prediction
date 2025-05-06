import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

# Load dataset
df = pd.read_csv('../vietnam_housing_dataset.csv', encoding='cp1252')

# Print initial info
print("\nInitial data info:")
print(df.info())
print("\nSample data:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for numeric columns
numeric_columns = ['Price', 'Area', 'Frontage', 'Access Road']
for col in numeric_columns:
    df = remove_outliers(df, col)

print("\nShape after removing outliers:", df.shape)

# Fill missing values using KNN Imputer for numeric columns
numeric_cols_for_impute = ['Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_cols_for_impute] = knn_imputer.fit_transform(df[numeric_cols_for_impute])

# Fill categorical missing values
df['House direction'] = df['House direction'].fillna('Không xác định')
df['Balcony direction'] = df['Balcony direction'].fillna('Không xác định')
df['Legal status'] = df['Legal status'].fillna('Khác')
df['Furniture state'] = df['Furniture state'].fillna('Không nội thất')

# Add interaction features
df['Area_Bathrooms'] = df['Area'] * df['Bathrooms']
df['Area_Floors'] = df['Area'] * df['Floors']
df['Frontage_AccessRoad'] = df['Frontage'] * df['Access Road']

# Add non-linear features
df['Area_Squared'] = df['Area'] ** 2
df['Log_Area'] = np.log1p(df['Area'])
df['Log_Price'] = np.log1p(df['Price'])

# Add derived features
df['Area_per_room'] = df['Area'] / (df['Bedrooms'] + df['Bathrooms'])
df['Rooms_per_floor'] = (df['Bedrooms'] + df['Bathrooms']) / df['Floors']
df['Total_Rooms'] = df['Bedrooms'] + df['Bathrooms']

# Print info after feature engineering
print("\nAfter feature engineering:")
print(df.info())
print("\nValue counts for categorical columns:")
for col in ['House direction', 'Balcony direction', 'Legal status', 'Furniture state']:
    print(f"\n{col}:")
    print(df[col].value_counts())

# Handle categorical variables properly for Vietnamese housing data
numeric_cols = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms',
                'Area_Bathrooms', 'Area_Floors', 'Frontage_AccessRoad', 
                'Area_Squared', 'Log_Area', 'Area_per_room', 'Rooms_per_floor', 'Total_Rooms']
categorical_cols = ['House direction', 'Balcony direction', 'Legal status', 'Furniture state']

# Define features and target
X = df[numeric_cols + categorical_cols].copy()
y = df['Log_Price'].values  # Use log-transformed price as target

# Print feature statistics
print("\nFeature statistics:")
print(X.describe())

# Create preprocessing pipeline with feature scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# Define improved models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.001),  # Reduced alpha for less regularization
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate each model
results = {}
best_model = None
best_score = float('-inf')

print("\n=== Model Comparison ===")
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions (in log space)
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    
    # Convert predictions back to original scale
    train_pred_original = np.expm1(train_pred)
    test_pred_original = np.expm1(test_pred)
    y_train_original = np.expm1(y_train)
    y_test_original = np.expm1(y_test)
    
    # Calculate metrics in original scale
    train_r2 = r2_score(y_train_original, train_pred_original)
    test_r2 = r2_score(y_test_original, test_pred_original)
    test_rmse = np.sqrt(mean_squared_error(y_test_original, test_pred_original))
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    
    results[name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'pipeline': pipeline
    }
    
    print(f"{name} Results:")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Print feature importance for tree-based models
    if hasattr(pipeline['regressor'], 'feature_importances_'):
        print("\nFeature Importance:")
        feature_names = (numeric_cols + 
                        [f"{col}_{val}" for col, vals in 
                         zip(categorical_cols, pipeline['preprocessor']
                             .named_transformers_['cat'].categories_) 
                         for val in vals])
        importances = pipeline['regressor'].feature_importances_
        importance_dict = dict(zip(feature_names, importances))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_importance[:10]:  # Show top 10 features
            print(f"{name}: {imp:.4f} ({imp*100:.2f}%)")
    
    # Print coefficients for linear models
    if hasattr(pipeline['regressor'], 'coef_'):
        print("\nTop 10 Feature Coefficients (Absolute Value):")
        feature_names = (numeric_cols + 
                        [f"{col}_{val}" for col, vals in 
                         zip(categorical_cols, pipeline['preprocessor']
                             .named_transformers_['cat'].categories_) 
                         for val in vals])
        coefficients = pipeline['regressor'].coef_
        coef_dict = dict(zip(feature_names, coefficients))
        sorted_coef = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, coef in sorted_coef[:10]:
            print(f"{name}: {coef:.4f}")
    
    # Update best model
    if test_r2 > best_score:
        best_score = test_r2
        best_model = pipeline

print("\n=== Best Model ===")
best_name = [name for name, res in results.items() 
            if res['pipeline'] is best_model][0]
print(f"Best model: {best_name}")
print(f"Test R²: {best_score:.4f}")

# Save best model
joblib.dump(best_model, 'house_price_model.pkl')

# Save feature information
feature_info = {
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'categorical_values': {
        'House direction': ["Không xác định", "Bắc", "Nam", "Đông", "Tây", "Đông Bắc", "Tây Bắc", "Đông Nam", "Tây Nam"],
        'Balcony direction': ["Không xác định", "Bắc", "Nam", "Đông", "Tây", "Đông Bắc", "Tây Bắc", "Đông Nam", "Tây Nam"],
        'Legal status': ["Sổ đỏ", "Sổ hồng", "Giấy tờ hợp lệ", "Đang chờ sổ", "Khác"],
        'Furniture state': ["Không nội thất", "Nội thất cơ bản", "Đầy đủ nội thất", "Cao cấp"]
    }
}
joblib.dump(feature_info, 'feature_info.pkl')

print("\nModel and feature information saved successfully.")