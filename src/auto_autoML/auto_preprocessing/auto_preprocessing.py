from mlbox.preprocessing import Reader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from auto_autoML.auto_preprocessing.auto_drift_detector import NumericalDrifter, CategoricalDrifter
from auto_autoML.auto_preprocessing.auto_cardinality_remover import HighCardinalityRemover
import pandas as pd
import numpy as np


def remove_ids(df):
    df = df.copy()
    cols_drop = [col for col in df.select_dtypes(['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
                 if df[col].diff().dropna().eq(1).all()]
    return df.drop(columns=cols_drop)


def train_test_split_wrapper(data_path, target_name, train, test, y_test):
    path = '/'.join(str(data_path).split('/')[:-1])
    train.to_csv(f'{path}/train.csv')
    test.to_csv(f'{path}/test.csv')

    paths = [f'{path}/train.csv',f'{path}/test.csv']

    rd = Reader(sep=",")
    df = rd.train_test_split(paths, target_name)

    if y_test.dtype == 'object':
        df['target_test'] = pd.Series(y_test)
        df['target_test'] = LabelEncoder().fit_transform(df['target_test'])
        return df

    elif y_test.dtype == 'int' or y_test.dtype == 'float':
        df['target_test'] = pd.Series(y_test)
        return df


def split_data(datapath, target_name, train_test_ratio=0.7, stratified=True):
    df = pd.read_csv(datapath).pipe(remove_ids)

    if stratified:
        X = df.drop(columns=target_name)
        y = df[target_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=train_test_ratio)
        train = pd.concat([X_train, y_train], axis=1)

        return train, X_test, y_test

    elif not stratified:

        train, test = np.split(df, [int(train_test_ratio * len(df))])
        y_test = test[target_name]
        X_test = test.drop(columns=[target_name])

        return train, X_test, y_test


def get_train_test_data(data_path, target_name, train_test_ratio=0.7, stratified=True):
    df_train, df_test, y_test = split_data(data_path, target_name, train_test_ratio, stratified)
    return train_test_split_wrapper(data_path, target_name, df_train, df_test, y_test)


class Preprocess:

    def __init__(self, numerical_drift_threshold=0.6, max_cardinality=0.9):
        self.numerical_drift_threshold = numerical_drift_threshold
        self.max_cardinality = max_cardinality

    def fit_transform(self, df):
        y_test = df['target_test']
        df = df.copy()
        df = NumericalDrifter(self.numerical_drift_threshold).fit_transform(df)
        df = CategoricalDrifter().fit_transform(df)
        df = HighCardinalityRemover(self.max_cardinality).fit_transform(df)
        df['target_test'] = y_test

        return df



