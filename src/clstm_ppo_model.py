"""
file name: clstm_ppo_model.ipynb
description: This notebook contains the implementation of the clstm_ppo_model described in the paper "A novel Deep Reinforcement Learning based automated stock trading system using cascaded LSTM networks" by Jie Zou et al.
author: Damiano Pasquini [pasquini.damiano00@gmail.com]
dataset citation: Dong, Z., Fan, X., & Peng, Z. (2024). FNSPID: A Comprehensive Financial News Dataset in Time Series. arXiv preprint arXiv:2402.06698.
license: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC-4.0) license
"""

import os, os.path
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from src.StockTradingEnv import StockTradingEnv
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# CUDA availability
def check_cuda(use_gpu=False):
    """
    This function checks the availability of CUDA and prints the version of PyTorch and CUDA, and the GPU name.
    :return: None
    """
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        if use_gpu:
            print("CUDA Version:", torch.version.cuda)
            print("GPU Name:", torch.cuda.get_device_name(0))
            torch.device("cuda")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print("Selected primary: GPU")
        else:
            torch.device("cpu")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("Selected primary: CPU")

def build_df(path, num_stocks):
    """
    This function builds a dataframe from the csv files in the path directory, with the first num_stocks.
    :param path: path to the directory containing the csv files
    :param num_stocks: number of stocks to consider, must be 5, 25 or 50
    :return: a dataframe containing the data from the csv files
    """
    if not os.path.exists(path):
        raise ValueError("The specified path does not exist.")
    if not os.path.isdir(path):
        raise ValueError("The specified path is not a directory.")
    if not os.listdir(path):
        raise ValueError("The specified directory is empty.")
    if not all([f.endswith('.csv') for f in os.listdir(path)]):
        raise ValueError("The specified directory contains files that are not CSV files.")
    if not num_stocks == 5 or num_stocks == 25 or num_stocks == 50:
        raise ValueError("The number of stocks must be 5, 25 or 50.")
    # Test csvs = 50
    names_50 = ['aal.csv', 'AAPL.csv', 'ABBV.csv', 'AMD.csv', 'amgn.csv', 'AMZN.csv', 'BABA.csv',
                'bhp.csv', 'bidu.csv', 'biib.csv', 'BRK-B.csv', 'C.csv', 'cat.csv', 'cmcsa.csv', 'cmg.csv',
                'cop.csv', 'COST.csv', 'crm.csv', 'CVX.csv', 'dal.csv', 'DIS.csv', 'ebay.csv', 'GE.csv',
                'gild.csv', 'gld.csv', 'GOOG.csv', 'gsk.csv', 'INTC.csv', 'KO.csv', 'mrk.csv', 'MSFT.csv',
                'mu.csv', 'nke.csv', 'nvda.csv', 'orcl.csv', 'pep.csv', 'pypl.csv', 'qcom.csv', 'QQQ.csv',
                'SBUX.csv', 'T.csv', 'tgt.csv', 'tm.csv', 'TSLA.csv', 'TSM.csv', 'uso.csv', 'v.csv', 'WFC.csv',
                'WMT.csv', 'xlf.csv']

    # Test csvs = 25
    names_25 = ['AAPL.csv', 'ABBV.csv', 'AMZN.csv', 'BABA.csv', 'BRK-B.csv', 'C.csv', 'COST.csv', 'CVX.csv',
                'DIS.csv', 'GE.csv', 'INTC.csv', 'MSFT.csv', 'nvda.csv', 'pypl.csv', 'QQQ.csv', 'SBUX.csv', 'T.csv',
                'TSLA.csv', 'WFC.csv', 'KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']

    # Test csvs = 5
    names_5 = ['KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']
    df = pd.DataFrame()
    stocks = names_5 if num_stocks == 5 else names_25 if num_stocks == 25 else names_50
    for stock in stocks:
        stock = pd.read_csv(path + '/' + stock)
        df = pd.concat([df, stock])
    df.sort_values(by='Date', inplace=True)
    return df

def feature_transformation(df):
    """
    This function is used to execute all the preprocessing steps over the features contained in the dataframe.
    These transformations must be applied only to train and validation sets, not to the test set.
    :param df: the dataframe
    :return: the transformed dataframe
    """
    df = df.copy()
    # Convert Date to datetime and set as index
    df["Date"] = pd.to_datetime(df["Date"])
    # df.set_index("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)  # Keep 'Date' as a column
    # Handle missing values
    df.ffill(inplace=True)
    # Standardize the data with StandardScaler (z-score)
    scaler = StandardScaler()
    cols_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj close', 'Volume', 'Sentiment_gpt', 'Scaled_sentiment']
    for col in cols_to_scale:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    # df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    # Normalize numerical columns
    scaler = MinMaxScaler()
    cols_to_normalize = ["Open", "High", "Low", "Close", 'Adj close', "Volume", "Sentiment_gpt", "Scaled_sentiment"]
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    print('Dataset preprocessing completed (standardization and normalization).')
    return df

def split_data(df, test_size=0.10, val_size=0.10):
    """
    Split the data into train, validation and test sets.
    :param df: the dataframe
    :param test_size: the size of the test set
    :param val_size: the size of the validation set
    :return: train, test, and validation sets
    """
    train_size = int(df.shape[0] * (1 - test_size - val_size))
    train = df[:train_size]
    test_size = int(df.shape[0] * test_size)
    test = df[train_size:train_size + test_size]
    val_size = int(df.shape[0] * val_size)
    val = df[train_size + test_size:train_size + test_size + val_size]
    return train, test, val

# CLSTM-PPO model

# Define LSTM-based feature extractor
class LSTMFeatureExtractor(nn.Module):
    """
    This class defines the LSTM-based feature extractor used in the CLSTM-PPO model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        This function performs the forward pass of the feature extractor.
        :param x: input tensor
        :return: features
        """
        lstm_out, _ = self.lstm(x)
        features = self.fc(lstm_out[:, -1, :])
        return features

# Define PPO Model with LSTM feature extractor
class CLSTM_PPO(nn.Module):
    """
    This class defines the CLSTM-PPO model.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CLSTM_PPO, self).__init__()
        self.feature_extractor = LSTMFeatureExtractor(state_dim, hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        This function performs the forward pass of the model.
        :param state: input tensor
        :return: action_probs, value
        """
        features = self.feature_extractor(state)
        action_probs = torch.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value

def main():
    # Load data
    dir_path = '../dataset/processed/data_for_lstm'
    if not os.path.exists(dir_path):
        raise ValueError("The specified path does not exist.")
    # CUDA availability (set True to use GPU)
    check_cuda(False)
    # Build dataframe
    data_frame = build_df(dir_path, 5)
    # Feature transformation
    data_frame = feature_transformation(data_frame)
    # Split data
    train_data, test_data, val_data = split_data(data_frame)
    # Environment Setup
    env = DummyVecEnv([lambda: StockTradingEnv(train_data)])
    # PPO model setup, to be run on CPU
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="../logs/ppo_logs/")
    # Training loop
    n_steps = 10000
    for i in range(n_steps):
        model.learn(total_timesteps=1000, reset_num_timesteps=False)
        if i % 10 == 0:
            model.save(f"../models/CLSTM_PPO_SENTIMENT/checkpoints/ppo_stock_{i}.zip")
            print(f"Checkpoint saved at step {i}")
    # Save final model
    model.save("../models/CLSTM_PPO_SENTIMENT/ppo_stock_final.zip")

if __name__ == '__main__':
    main()
