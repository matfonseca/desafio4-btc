from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

# All sklearn Transforms must have the `transform` and `fit` methods
class Smote(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, data_to_smote):
        data = pd.DataFrame.from_records(data= data_to_smote,columns= self.columns)
        # for reproducibility purposes
        seed = 100
        # SMOTE number of neighbors
        k = 1
        # make a new df made of all the columns, except the target class
        X = data.drop('OBJETIVO', axis=1)
        y = data["OBJETIVO"]
        sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)
        X_res, y_res = sm.fit_resample(X, y)
        # Primero copiamos el dataframe de datos de entrada 'X'
        X_res["OBJETIVO"] = y_res
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        return X_res