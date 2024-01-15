import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from deteccion_de_autismo.interface.main_local import load_model

app = FastAPI()
#app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {
    'greeting': 'Hello'
}
