import os
import logging
from pathlib import Path

import pandas as pd
from joblib import load

# load .env content


PROJECT_ROOT = Path(__file__).resolve().parent.parent 
LOG_PATH = PROJECT_ROOT / "logs" / "app.log"
MODEL_PATH = PROJECT_ROOT / "model_dir" / "HeartDiseasePrediction.joblib"
DATASET_PATH = (
    PROJECT_ROOT / "Data" / "heart.csv"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)]
)

# load the trained model only once
trained_model = load(MODEL_PATH)
logging.info('Model Loaded successfully.')

def predict(input_data: dict):
    
    df = pd.DataFrame([input_data])
    
    # get predicted class
    prediction = int(trained_model.predict(df)[0])
    
    # get probability of the prediction
    probab = float(trained_model.predict_proba(df)[0][1])
    
    logging.info(f'model predicted: {prediction} and w/ probability of {probab}')
    
    return {
        "prediction": prediction,
        "probability": probab
    }
    
# example usage:
# input_data = {
#     "age"
# }
    
    
    
