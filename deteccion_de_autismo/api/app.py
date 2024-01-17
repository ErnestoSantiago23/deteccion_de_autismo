import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from deteccion_de_autismo.interface.main_local import load_model
from pydantic import BaseModel

app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class AutismPredictionRequest(BaseModel):
    Months_encoder: int
    Sex: int
    Ethnicity_encoder: int
    Family_mem_with_ASD: int
    A1: int
    A2: int
    A3: int
    A4: int
    A5: int
    A6: int
    A7: int
    A8: int
    A9: int
    A10: int



@app.post("/predict")
def predict(request: AutismPredictionRequest):
    data = np.array([[
        request.Months_encoder,
        request.Sex,
        request.Ethnicity_encoder,
        request.Family_mem_with_ASD,
        request.A1,
        request.A2,
        request.A3,
        request.A4,
        request.A5,
        request.A6,
        request.A7,
        request.A8,
        request.A9,
        request.A10
    ]])

    model = app.state.model
    prediccion = model.predict(data)
    proba = model.predict_proba(data)
    return {
        'prediccion': int(prediccion[0]),
        'probability': proba[0].tolist()
    }


@app.get("/")
def root():
    return {
    'greeting': 'Hello, welcome to Spectruminsight API'
}
