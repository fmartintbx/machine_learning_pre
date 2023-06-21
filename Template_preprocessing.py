import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Tratamiento de los Nas
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Supongamos que tu matriz de características se llama X codificar datos categoricos
labelencoder_X = LabelEncoder()
label_encoder_y = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
y_encoded = label_encoder_y.fit_transform(y)

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],  # Columna 0 es categórica
    remainder='passthrough'  # Mantener las otras columnas sin cambios
)
X = ct.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)