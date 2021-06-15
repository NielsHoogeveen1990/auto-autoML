import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from mlbox.preprocessing import Drift_thresholder


class NumericalDrifter(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.dft = Drift_thresholder(threshold=self.threshold)

    def fit(self, df):
        return self

    def transform(self, df):
        return self.dft.fit_transform(df)


class CategoricalDrifter(BaseEstimator, TransformerMixin):

    def fit(self, df):
        df_temp = self.generate_temp_dataframe(df['train'], df['test'])
        self.drifted_cat_cols_ = self.chi_square_test(df_temp)
        return self

    def transform(self, df):
        df['train'].drop(columns=self.drifted_cat_cols_)
        df['test'].drop(columns=self.drifted_cat_cols_)

        return df

    @staticmethod
    def get_categorical_columns(X):
        return list(X.select_dtypes(include=['category', 'object']).columns)

    @staticmethod
    def generate_temp_dataframe(train, test):
        train, test = train.copy(), test.copy()

        train = train.assign(type_df='train')
        test = test.assign(type_df='test')

        return pd.concat([train, test])

    def chi_square_test(self, df):
        cat_cols = [col for col in self.get_categorical_columns(df)]
        cat_cols.remove('type_df')
        drift_cols = []

        for col in cat_cols:
            c, p, dof, expected = chi2_contingency(pd.crosstab(df[col], df['type_df']))

            if p < 0.05:
                drift_cols.append(col)

        return drift_cols

