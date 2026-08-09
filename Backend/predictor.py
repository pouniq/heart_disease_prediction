import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load

# load .env content
load_dotenv()


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
DATASET_PATH = (
    PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
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
    
    
    
