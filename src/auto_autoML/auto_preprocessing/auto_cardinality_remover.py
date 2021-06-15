from sklearn.base import TransformerMixin, BaseEstimator


class HighCardinalityRemover(TransformerMixin, BaseEstimator):
    def __init__(self, max_cardinality=0.9):
        self.max_cardinality = max_cardinality

    def fit(self, df):
        X = df['train']

        columns = self.get_categorical_columns(X)

        self.columns_to_drop_ = X[columns].nunique()[lambda x: (x / len(df)) < self.max_cardinality].index
        return self

    def transform(self, df):
        df = df.copy()
        df['train'].drop(columns=self.columns_to_drop_)
        df['test'].drop(columns=self.columns_to_drop_)

        return df

    @staticmethod
    def get_categorical_columns(X):
        return list(X.select_dtypes(include=['category', 'object']).columns)