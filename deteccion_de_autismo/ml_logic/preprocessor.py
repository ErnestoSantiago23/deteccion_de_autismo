import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def preprocess_and_econding(data):
    """
    This function takes a dataframe and process the features in the first place
    and then does an ordinal encoding
    """

    # PREPROCESS DATA
    drop_columns = ['Qchat-10-Score', 'Jaundice', 'Who completed the test']

    data = data.drop(drop_columns, axis=1)

    data['Sex'] = data['Sex'].map({'f': 0, 'm': 1})

    data['Family_mem_with_ASD'] = data['Family_mem_with_ASD'].str.strip().map({'no': 0, 'yes': 1})

    data['Class/ASD Traits '] = data['Class/ASD Traits '].str.strip().map({'No': 0, 'Yes': 1})

    data.rename(columns={'Class/ASD Traits ': 'Class/ASD Traits'}, inplace=True)

    bins = [12, 18, 24, 30, 37]
    labels = [0, 1, 2, 3]

    data['Months_encoder'] = pd.cut(data['Age_Mons'], bins=bins, labels=labels, right=False)

    data = data.drop(['Age_Mons'], axis=1)

    # ENCODING DATA
    ordinal_encoder = OrdinalEncoder()

    data["Ethnicity_encoder"] = ordinal_encoder.fit_transform(data[["Ethnicity"]])

    data = data.drop(['Ethnicity'], axis=1)

    column_order = ['Months_encoder', 'Sex', 'Ethnicity_encoder', 'Family_mem_with_ASD', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Class/ASD Traits',]
    data = data[column_order]

    return data
