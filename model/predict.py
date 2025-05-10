import joblib
import numpy as np
import pandas as pd
import os

def predict_price(input_data):
    """
    Predict house price based on input features
    
    Args:
        input_data: List of features in the correct order
        
    Returns:
        float: Predicted price in VND
    """
    try:
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, 'house_price_model.pkl')
        feature_info_path = os.path.join(base_dir, 'feature_info.pkl')
        
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
        df['Log_Area'] = np.log1p(df['Area'])
        df['Area_per_room'] = df['Area'] / (df['Bedrooms'] + df['Bathrooms'])
        df['Total_Rooms'] = df['Bedrooms'] + df['Bathrooms']
        
        # Make prediction (model will handle log transformation internally)
        prediction = model.predict(df)[0]
        
        # Convert from log scale back to original scale
        final_prediction = np.expm1(prediction)
        
        return final_prediction
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise
