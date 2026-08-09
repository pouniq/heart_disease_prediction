from fastapi import FastAPI
from pydantic import BaseModel

from Backend.predictor import predict

app = FastAPI(
    title= 'heart disease prediction API',
    version="1.0.0"
    
)

# input schema

class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
    
@app.get("/health")
def health_check():
    return {"status": 'ok'}

# heart disease prediction endpoint
@app.post("/predict_heart_disease")
def predict_heart_disease(input_data: HeartDiseaseInput):
    input_data = input_data.model_dump()
    result = predict(input_data=input_data)
    return {
        "prediction": result['prediction'],
        'probability': result['probability'],
        'diagnosis': (
            "heart disease detected"
            if result['prediction'] == 1
            else "No heart disease detected"
        )
    }
