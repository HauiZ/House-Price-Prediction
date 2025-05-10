import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import os

def train_decision_tree_model(data_path, output_dir='.'):
    """
    Train a Decision Tree Regressor model for house price prediction
    
    Args:
        data_path: Path to the Vietnam housing dataset CSV file
        output_dir: Directory to save the model and feature info
        
    Returns:
        tuple: (model, feature_info)
    """
    print("Loading data...")
    # Try different encodings to handle Vietnamese characters properly
    try:
        # Try utf-8 first (most common encoding for international text)
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # Try utf-8-sig (UTF-8 with BOM)
            df = pd.read_csv(data_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            try:
                # Try utf-16
                df = pd.read_csv(data_path, encoding='utf-16')
            except Exception:
                # Last resort, let pandas detect encoding
                df = pd.read_csv(data_path)
    
    print("\nInitial data info:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
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
    
    print("\nRemoving outliers...")
    # Remove outliers for numeric columns
    numeric_columns = ['Price', 'Area', 'Frontage', 'Access Road']
    for col in numeric_columns:
        df = remove_outliers(df, col)
    
    print(f"Shape after removing outliers: {df.shape}")
    
    # Fill missing values using KNN Imputer for numeric columns
    print("\nFilling missing values...")
    numeric_cols_for_impute = ['Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
    knn_imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols_for_impute] = knn_imputer.fit_transform(df[numeric_cols_for_impute])
    
    # Fill categorical missing values
    df['House direction'] = df['House direction'].fillna('Không xác định')
    df['Balcony direction'] = df['Balcony direction'].fillna('Không xác định')
    df['Legal status'] = df['Legal status'].fillna('Khác')
    df['Furniture state'] = df['Furniture state'].fillna('Không nội thất')
    
    print("\nCreating derived features...")
    # Add interaction features
    df['Area_Bathrooms'] = df['Area'] * df['Bathrooms']
    df['Area_Floors'] = df['Area'] * df['Floors']
    df['Frontage_AccessRoad'] = df['Frontage'] * df['Access Road']
    
    # Add non-linear features
    df['Area_Squared'] = df['Area'] ** 2
    df['Log_Area'] = np.log1p(df['Area'])
    df['Log_Price'] = np.log1p(df['Price'])  # Log transform the target variable
    
    # Add derived features
    df['Area_per_room'] = df['Area'] / (df['Bedrooms'] + df['Bathrooms'])
    df['Rooms_per_floor'] = (df['Bedrooms'] + df['Bathrooms']) / df['Floors']
    df['Total_Rooms'] = df['Bedrooms'] + df['Bathrooms']
    
    # Handle categorical variables properly for Vietnamese housing data
    numeric_cols = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms',
                   'Area_Bathrooms', 'Area_Floors', 'Frontage_AccessRoad', 
                   'Area_Squared', 'Log_Area', 'Area_per_room', 'Rooms_per_floor', 'Total_Rooms']
    categorical_cols = ['House direction', 'Balcony direction', 'Legal status', 'Furniture state']
    
    # Define features and target
    X = df[numeric_cols + categorical_cols].copy()
    y = df['Log_Price'].values  # Use log-transformed price as target
    
    # Create preprocessing pipeline with feature scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    # Define Decision Tree model with hyperparameters
    decision_tree = DecisionTreeRegressor(
        max_depth=10,            # Maximum depth of the tree
        min_samples_split=5,     # Minimum samples required to split an internal node
        min_samples_leaf=4,      # Minimum samples required to be at a leaf node
        random_state=42          # For reproducibility
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', decision_tree)
    ])
    
    # Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining Decision Tree model...")
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
    
    print("\nDecision Tree Results:")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Print feature importance
    feature_names = (numeric_cols + 
                    [f"{col}_{val}" for col, vals in 
                     zip(categorical_cols, pipeline['preprocessor']
                         .named_transformers_['cat'].categories_) 
                     for val in vals])
    importances = pipeline['regressor'].feature_importances_
    importance_dict = dict(zip(feature_names, importances))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Feature Importance:")
    for name, importance in sorted_importance[:10]:
        print(f"{name}: {importance:.4f} ({importance*100:.2f}%)")
    
    # Save model and feature information
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'house_price_model_dt.pkl')
    feature_info_path = os.path.join(output_dir, 'feature_info_dt.pkl')
    
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")
    
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
    joblib.dump(feature_info, feature_info_path)
    print(f"Feature information saved to {feature_info_path}")
    
    return pipeline, feature_info


def predict_price(input_data, model_path='house_price_model_dt.pkl', feature_info_path='feature_info_dt.pkl'):
    """
    Predict house price based on input features
    
    Args:
        input_data: List of features in the correct order
        model_path: Path to the saved model
        feature_info_path: Path to the saved feature information
        
    Returns:
        float: Predicted price in VND
    """
    try:
        # Load model and feature information
        model = joblib.load(model_path)
        feature_info = joblib.load(feature_info_path)
        
        # Get numeric and categorical columns
        numeric_cols = feature_info['numeric_cols']
        categorical_cols = feature_info['categorical_cols']
        
        # Create DataFrame with numeric data first
        basic_numeric_cols = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']
        df = pd.DataFrame([input_data[:len(basic_numeric_cols)]], columns=basic_numeric_cols)
        
        # Add categorical data
        start_idx = len(basic_numeric_cols)
        for col in categorical_cols:
            values = feature_info['categorical_values'][col]
            n_values = len(values)
            one_hot = input_data[start_idx:start_idx + n_values]
            if 1 in one_hot:
                selected_idx = one_hot.index(1)
                df[col] = values[selected_idx]
            else:
                df[col] = values[0]  # Default to first value if none selected
            start_idx += n_values
        
        # Add derived features
        df['Area_Bathrooms'] = df['Area'] * df['Bathrooms']
        df['Area_Floors'] = df['Area'] * df['Floors']
        df['Frontage_AccessRoad'] = df['Frontage'] * df['Access Road']
        df['Area_Squared'] = df['Area'] ** 2
        df['Log_Area'] = np.log1p(df['Area'])
        df['Area_per_room'] = df['Area'] / (df['Bedrooms'] + df['Bathrooms'])
        df['Rooms_per_floor'] = (df['Bedrooms'] + df['Bathrooms']) / df['Floors']
        df['Total_Rooms'] = df['Bedrooms'] + df['Bathrooms']
        
        # Make prediction (model will handle log transformation internally)
        prediction = model.predict(df)[0]
        
        # Convert from log scale back to original scale
        final_prediction = np.expm1(prediction)
        
        return final_prediction
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise


def serve_prediction(args):
    """
    Command-line interface for house price prediction
    
    Args:
        args: List of string arguments from command line
        
    Returns:
        float: Predicted price in VND
    """
    try:
        if len(args) < 2:
            print("Error: No input features provided")
            return 1
        
        # Convert all arguments to float
        input_data = []
        for arg in args[1:]:
            try:
                value = float(arg)
                input_data.append(value)
            except ValueError:
                print(f"Error: Invalid input value: {arg}")
                return 1
        
        # Predict price
        prediction = predict_price(input_data)
        
        # Print prediction
        print(prediction)
        return 0
        
    except Exception as e:
        print(f"Error in prediction service: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # If first argument is "train", train the model
        data_path = sys.argv[2] if len(sys.argv) > 2 else "../vietnam_housing_dataset.csv"
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
        train_decision_tree_model(data_path, output_dir)
    else:
        # Otherwise, serve predictions
        exit(serve_prediction(sys.argv))
