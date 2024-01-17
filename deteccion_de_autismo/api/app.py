import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from deteccion_de_autismo.interface.main_local import load_model


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

@app.get("/predict")
def predict(Months_encoder,
            Sex,
            Ethnicity_encoder,
            Family_mem_with_ASD,
            A1, A2, A3, A4, A5,
            A6, A7, A8, A9, A10
            ):

    data = np.array([[
        Months_encoder,
        Sex,
        Ethnicity_encoder,
        Family_mem_with_ASD,
        A1,
        A2,
        A3,
        A4,
        A5,
        A6,
        A7,
        A8,
        A9,
        A10
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
