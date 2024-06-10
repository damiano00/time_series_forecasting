import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import yfinance as yf
import random


def analyze_dataset(args):
    """
    For each of the files in the args in input, it will analyze the dataset and return the stats
    :param args: file path or files paths to analyze
    :return: duplicated data, missing data, data types, and data distribution
    """
    for arg in args:
        if arg.endswith('.csv'):
            df = pd.read_csv(arg)
        elif arg.endswith('.xlsx'):
            df = pd.read_excel(arg)
        else:
            raise ValueError('File format not supported')
        duplicated_data = df.duplicated().sum()
        missing_data = df.isnull().sum()
        data_types = df.dtypes
        data_distribution = df.describe()
        print(f'Duplicated data: {duplicated_data}')
        print(f'Missing data: {missing_data}')
        print(f'Data types: {data_types}')
        print(f'Data distribution: {data_distribution}')
        for column in df.columns:
            if df[column].dtype == 'object':
                continue
            plt.hist(df[column], bins=len(df["label"].unique()))
            plt.title(f'Distribution of {column} in {arg}')
            plt.show()





