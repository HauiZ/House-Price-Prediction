import sys
from predict import predict_price

def main():
    try:
        # Get input arguments from command line
        # Skip the first argument (script name)
        if len(sys.argv) < 2:
            print("Error: No input features provided")
            sys.exit(1)
        
        # Convert all arguments to float
        input_data = []
        for arg in sys.argv[1:]:
            try:
                value = float(arg)
                input_data.append(value)
            except ValueError:
                # For binary features (0 or 1), they should already be converted to float
                print(f"Error: Invalid input value: {arg}")
                sys.exit(1)
        
        # Predict price
        prediction = predict_price(input_data)
        
        # Print only the prediction (will be captured by Node.js)
        print(prediction)
        
    except Exception as e:
        print(f"Error in prediction service: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()