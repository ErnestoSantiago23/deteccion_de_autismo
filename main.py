import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deteccion_de_autismo.ml_logic.preprocessor import preprocess_and_econding
from deteccion_de_autismo.ml_logic.model import train_model

from sklearn.ensemble import RandomForestClassifier

path = 'data/csv/Toddler Autism dataset July 2018.csv'
data = pd.read_csv(path, header = 0, index_col = 0)

data = preprocess_and_econding(data)

X = data.drop('Class/ASD Traits', axis=1)
y = data['Class/ASD Traits']

model = RandomForestClassifier(random_state=42)

y_pred = train_model(model, X, y)
