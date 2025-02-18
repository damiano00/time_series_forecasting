import os
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

class DataProcessor:

    def __init__(self, dir_path, seq_len, test_size, cols_to_scale, cols_to_normalize, batch_size):
        self.dir_path = dir_path
        self.seq_len = seq_len  # Sequence length
        self.train_size = 1 - test_size
        self.test_size = test_size
        self.cols_to_scale = cols_to_scale
        self.cols_to_normalize = cols_to_normalize
        self.batch_size = batch_size
        self.df = None
        self.train_data = None
        self.test_data = None
        self.feature_columns = None
        self.label_columns = ['Close', 'Adj close']

    def split_df(self):
        if self.df is None:
            raise ValueError("The dataframe is empty.")
        self.train_data = self.df[:int(len(self.df) * self.train_size)]
        self.test_data = self.df[int(len(self.df) * self.train_size):]

    def csv_to_df(self, stocks_list):
        self.df = pd.DataFrame()
        for stock in stocks_list:
            stock_data = pd.read_csv(os.path.join(self.dir_path, stock))
            self.df = pd.concat([self.df, stock_data])
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.ffill(inplace=True)
        self.df.sort_values(by='Date', inplace=True)
        self.transform_dates(self.df)
        self.feature_columns = [col for col in self.df.columns if col not in self.label_columns]

    @staticmethod
    def transform_dates(df):
        df["Date"] = pd.to_datetime(df["Date"])
        df.reset_index(drop=True, inplace=True)
        df['month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['Date'].dt.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['Date'].dt.day / 31)
        df['weekday'] = df['Date'].dt.weekday
        df.drop(columns=['Date'], inplace=True)

    def feature_transformation(self, df, is_train=True):
        if self.cols_to_scale:
            scaler = StandardScaler()
            for col in self.cols_to_scale:
                if col in df.columns:
                    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        if self.cols_to_normalize:
            scaler = MinMaxScaler()
            for col in self.cols_to_normalize:
                if col in df.columns:
                    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    def _create_dataset(self, dataset):
        X, y = [], []
        for i in range(len(dataset) - self.seq_len):
            X.append(dataset[self.feature_columns].iloc[i:(i + self.seq_len)].values)
            y.append(dataset[self.label_columns].iloc[i + self.seq_len].values)
        return np.array(X), np.array(y)

    def get_train_loader(self):
        X_train, y_train = self._create_dataset(self.train_data)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        train_dataset = TensorDataset(X_train, y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_test_loader(self):
        X_test, y_test = self._create_dataset(self.test_data)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        test_dataset = TensorDataset(X_test, y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
