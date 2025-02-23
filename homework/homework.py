#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
import joblib
import gzip
import json
from pathlib import Path

# Paso 1: Preprocesar datos
def preprocess_data(df):
    df = df.copy()
    df["Age"] = 2021 - df["Year"]
    df = df.drop(columns=["Year", "Car_Name"])
    x = df.drop(columns=["Present_Price"])  
    y = df["Present_Price"] 
    return df, x, y

# Cargar datos
train_df = pd.read_csv('files/input/train_data.csv.zip', compression='zip')
test_df = pd.read_csv('files/input/test_data.csv.zip', compression='zip')
print(train_df.columns)
print(test_df.columns)
train_df, X_train, y_train = preprocess_data(train_df)
test_df, X_test, y_test = preprocess_data(test_df)

# Paso 3: Crear pipeline
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
numerical_features = [col for col in X_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical_features),
    ('scaler', MinMaxScaler(), numerical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_regression)),
    ('classifier', LinearRegression())
])

# Paso 4: Optimizar hiperparámetros
param_grid = {'feature_selection__k': range(1,25),
               'classifier__fit_intercept':[True,False],
        'classifier__positive':[True,False]}  

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1,
    refit=True
)
    
estimator = grid_search.fit(X_train, y_train)


# Paso 5: Guardar modelo
Path('files/models').mkdir(parents=True, exist_ok=True)
with gzip.open('files/models/model.pkl.gz', 'wb') as f:
    joblib.dump(estimator, f)

# Paso 6: Calcular métricas
def compute_metrics(model, X, y, dataset_type):
    y_pred = model.predict(X)
    return {
        'type': 'metrics',
        'dataset': dataset_type,
        'r2': float(r2_score(y, y_pred)),
        'mse': float(mean_squared_error(y, y_pred)),
        'mad': float(median_absolute_error(y, y_pred))
    }

train_metrics = compute_metrics(estimator, X_train, y_train, 'train')
test_metrics = compute_metrics(estimator, X_test, y_test, 'test')

# Guardar métricas
Path('files/output').mkdir(parents=True, exist_ok=True)
with open('files/output/metrics.json', 'w') as f:
    f.write(json.dumps(train_metrics) + '\n')
    f.write(json.dumps(test_metrics) + '\n')
