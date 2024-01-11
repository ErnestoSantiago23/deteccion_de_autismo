import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from deteccion_de_autismo.ml_logic.preprocessor import preprocess_and_econding

path = 'data/csv/Toddler Autism dataset July 2018.csv'

data = pd.read_csv(path, header = 0, index_col = 0)
data = preprocess_and_econding(data)

X = data.drop('Class/ASD Traits', axis=1)
y = data['Class/ASD Traits']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    modelo_entrenado = model.fit(X_train, y_train)

    return modelo_entrenado



def load_model():
    my_model = pickle.load(open("model.pkl","rb"))
    return my_model


def predict(model, X_test):
    result = model.predict(X_test)
    return result

if __name__ == '__main__':
    model = train_model(X_train, y_train)
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    predict(model, X_test)
