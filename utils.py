"""
utils.py
--------
Transformadores customizados para a pipeline de pré-processamento do
modelo preditivo de obesidade.

Todos os transformadores seguem a interface do scikit-learn
(BaseEstimator + TransformerMixin), o que permite integrá-los
diretamente em um objeto Pipeline e serializá-los via joblib.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


class DropFeatures(BaseEstimator, TransformerMixin):
    """
    Remove colunas especificadas do DataFrame.

    Utilizado para excluir variáveis que causariam data leakage
    ou que não devem ser usadas como features pelo modelo
    (ex.: 'Weight' e 'Height', que derivam diretamente do target via IMC).

    Parâmetros
    ----------
    feature_to_drop : list[str]
        Lista com os nomes das colunas a serem removidas.
    """

    def __init__(self, feature_to_drop):
        self.feature_to_drop = feature_to_drop

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if set(self.feature_to_drop).issubset(df.columns):
            return df.drop(self.feature_to_drop, axis=1)
        print('Uma ou mais features não estão no DataFrame')
        return df


class MinMax(BaseEstimator, TransformerMixin):
    """
    Aplica normalização Min-Max nas colunas numéricas especificadas.

    O scaler é fitado apenas nos dados de treino (método fit) e
    reutilizado nas transformações subsequentes (método transform),
    evitando data leakage.

    Parâmetros
    ----------
    min_max_scaler : list[str]
        Lista com os nomes das colunas numéricas a serem normalizadas.
    """

    def __init__(self, min_max_scaler):
        self.min_max_scaler = min_max_scaler

    def fit(self, df, y=None):
        # Instancia e fita o scaler apenas com os dados de treino
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(df[self.min_max_scaler])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.min_max_scaler] = self.scaler_.transform(df[self.min_max_scaler])
        return df


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Aplica encoding ordinal nas colunas categóricas especificadas.

    Permite definir a ordem das categorias explicitamente, o que é
    importante para variáveis com ordem semântica, como:
        no < Sometimes < Frequently < Always  (CAEC, CALC)

    Valores desconhecidos em produção são codificados como -1.

    Parâmetros
    ----------
    columns : list[str]
        Lista com os nomes das colunas a serem encodadas.
    categories : 'auto' ou list[list]
        Ordem das categorias por coluna. Use 'auto' para variáveis
        sem ordem relevante (binárias como yes/no, Male/Female).
        Para variáveis ordinais, passe uma lista de listas com a
        ordem correta — uma lista por coluna.
    """

    def __init__(self, columns, categories='auto'):
        self.columns = columns
        self.categories = categories

    def fit(self, X, y=None):
        self.encoder = OrdinalEncoder(
            categories=self.categories,
            handle_unknown='use_encoded_value',
            unknown_value=-1  # valores não vistos no treino recebem -1
        )
        self.encoder.fit(X[self.columns])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.columns] = self.encoder.transform(X_copy[self.columns])
        return X_copy


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Aplica One-Hot Encoding nas colunas categóricas especificadas.

    As colunas originais são substituídas pelas colunas binárias geradas.
    Categorias desconhecidas em produção são ignoradas (todos zeros).

    Utilizado para a variável 'MTRANS' (meio de transporte), que não
    possui ordem semântica entre suas categorias.

    Parâmetros
    ----------
    columns : list[str]
        Lista com os nomes das colunas a serem encodadas.
    """

    def __init__(self, columns):
        self.columns = columns
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names = None

    def fit(self, X, y=None):
        self.encoder.fit(X[self.columns])
        # Armazena os nomes das colunas geradas para uso no transform
        self.feature_names = self.encoder.get_feature_names_out(self.columns)
        return self

    def transform(self, X):
        X_copy = X.copy()
        encoded_data = self.encoder.transform(X_copy[self.columns])

        # Cria DataFrame com as colunas encodadas mantendo o índice original
        encoded_df = pd.DataFrame(encoded_data, columns=self.feature_names, index=X_copy.index)

        # Remove as colunas originais e concatena as novas colunas binárias
        X_copy = X_copy.drop(columns=self.columns)
        return pd.concat([X_copy, encoded_df], axis=1)

