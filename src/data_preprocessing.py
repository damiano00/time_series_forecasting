import csv
import pandas
import pandas as pd
from dotenv import load_dotenv, dotenv_values
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import yfinance as yf
import random


# Load CSV file
def csv_to_df(path):
    """
    This function will load the CSV file from the given path
    :param path: path to the CSV file
    :return: Pandas DataFrame
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("The file was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("No data found in the file.")
        return None
    except pd.errors.ParserError:
        print("Error parsing the file.")
        return None


def get_stats(df):
    """
    This function will generate statistics for the given DataFrame and will return them as a dictionary, printing them
    :param df: DataFrame
    :return: dictionary containing statistics, or None if the DataFrame is empty
    """
    stats = {'shape': df.shape, 'columns': df.columns.tolist(), 'dtypes': df.dtypes.to_dict(),
             'missing_values': df.isnull().sum().to_dict(), 'describe': df.describe().to_dict()}
    for key, value in stats.items():
        print(key, ":", value, '\n')
    return stats


def chatgpt_sentiment(path, system_instructions, prompt):
    # TODO: implement
    """
    This function will use the chatGPT sentiment analysis to analyze the sentiment of the text in the file
    :param path: path to the csv file
    :return: sentiment analysis
    """
    with open(path, 'r') as file:
        text = file.read()
    # sentiment analysis
    system_instructions = "Forget all your previous instructions. You are a financial expert with stock recommendation experience. Based on a specific stock, score for range from 1 to 5, where 1 is negative, 2 is somewhat negative, 3 is neutral, 4 is somewhat positive, 5 is positive. 10 summarized news will be passed in each time, you will give score in format as shown below in the response from assistant."
    for line in text.split('\n'):
        symbol = line.split(',')[0]
        promt = f'News to Stock Symbol -- {symbol}: Apple (AAPL) increase 22% ###'


def reduce_dataset(path, output_path, no_lines_to_skip):
    """
    This function will reduce the dataset by the given percentage and save it to the output path
    :param path: path to the input CSV file
    :param output_path: path to the output CSV file
    :param no_lines_to_skip: percentage of the dataset to keep
    :return: None
    """
    # assert 1 <= no_lines_to_skip <= 10, "Percentage must be between 1 and 100."
    df = pd.read_csv(path, skiprows=lambda x: x % no_lines_to_skip != 0)
    df.to_csv(output_path, index=False)
    return df

