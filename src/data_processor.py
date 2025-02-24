import datetime
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class DataProcessor:
    def __init__(self, dir_path: str, seq_len: int, test_size: float, cols_to_normalize: list[str], batch_size: int,
                 smoothing_window_size: int, feature_columns: list[str], label_columns: list[str]):
        self.dir_path = dir_path
        self.df = None
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.smoothing_window_size = smoothing_window_size
        self.train_size = 1 - test_size
        self.test_size = test_size
        self.train_data = None
        self.test_data = None
        self.cols_to_normalize = cols_to_normalize
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_csv(self, stocks_list: list[str], make_date_index: bool = True):
        """
        Read the CSV files and concatenate them into a single dataframe.
        :param stocks_list: List of stock CSV files
        :param make_date_index: Whether to set the 'Date' column as the index
        """
        df_list = []
        for stock in stocks_list:
            stock_data = pd.read_csv(os.path.join(self.dir_path, stock) if self.dir_path else stock)
            df_list.append(stock_data)

        self.df = pd.concat(df_list, ignore_index=True)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)  # Backward fill as well

        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"], errors='coerce')
            self.df.dropna(subset=["Date"], inplace=True)
            self.df.sort_values(by="Date", inplace=True)
            if make_date_index:
                self.df.set_index("Date", inplace=True)

        self.feature_columns = [col for col in self.df.columns if col not in self.label_columns]

    def split_df(self):
        """Split the dataframe into training and testing sets."""
        if self.df is None or self.df.empty:
            raise ValueError("The dataframe is empty.")
        split_index = int(len(self.df) * self.train_size)
        self.train_data = self.df.iloc[:split_index].copy()
        self.test_data = self.df.iloc[split_index:].copy()

    def normalize_features(self):
        """Normalize specified columns using MinMaxScaler."""
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data not split. Run `split_df()` first.")

        missing_cols = [col for col in self.cols_to_normalize if col not in self.train_data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for normalization: {missing_cols}")

        self.scaler.fit(self.train_data[self.cols_to_normalize])

        self.train_data[self.cols_to_normalize] = self.scaler.transform(self.train_data[self.cols_to_normalize])
        self.test_data[self.cols_to_normalize] = self.scaler.transform(self.test_data[self.cols_to_normalize])

        print("Feature normalization complete.")

    def _create_sequences(self, dataset: pd.DataFrame):
        """Create sequences for LSTM input."""
        X, y = [], []
        dataset_values = dataset[self.feature_columns].values
        label_values = dataset[self.label_columns].values

        for i in range(len(dataset) - self.seq_len):
            X.append(dataset_values[i:(i + self.seq_len)])
            y.append(label_values[i + self.seq_len])

        return np.array(X), np.array(y)

    def get_data_loader(self, is_train: bool = True):
        """Create a PyTorch DataLoader."""
        dataset = self.train_data if is_train else self.test_data
        if dataset is None:
            raise ValueError("Data has not been split. Run `split_df()` first.")

        X_data, y_data = self._create_sequences(dataset)
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.float32)
        dataset_tensor = TensorDataset(X_tensor, y_tensor)

        return DataLoader(dataset_tensor, batch_size=self.batch_size, shuffle=False)
