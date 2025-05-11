import sys
import os
import logging
from predict import predict_price

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        input_data = []
        for arg in sys.argv[1:]:
            try:
                value = float(arg)
                input_data.append(value)
            except ValueError:
                sys.exit(1)
            
        prediction = predict_price(input_data)
        
        print(prediction)
        
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()