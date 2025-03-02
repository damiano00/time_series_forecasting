import os
import pandas as pd
import warnings


def generate_dataset(path, save_path, min_start_date, max_end_date, num_stocks=30, sentiment=False):
    # Get the list of stocks that have data in the specified time range
    stocks = get_stocks_with_time_range(path, min_start_date, max_end_date, num_stocks)
    # generate a new csv folder for the selected stocks
    os.makedirs(f'{save_path}/data_{num_stocks}_{"sentiment" if sentiment else "no_sentiment"}', exist_ok=True)
    for stock in stocks:
        data = pd.read_csv(os.path.join(path, stock))
        df = pd.DataFrame(data)
        if sentiment:
            # take the sentiment data from another folder, taking the same csv file name
            # TODO: tofix
            sentiment_data = pd.read_csv(f'data/{stock}')[['date', 'scaled_sentiment']]
            sentiment_df = pd.DataFrame(sentiment_data)
            df = pd.merge(df, sentiment_df, on='date', how='inner')
        # convert df['date'] to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        # remove the rows that are out of the time range
        df = df[(df['date'] >= min_start_date) & (df['date'] <= max_end_date)]
        # check if the date is in the range
        df_start_date = pd.to_datetime(df['date'], format='%Y-%m-%d').min()
        df_end_date = pd.to_datetime(df['date'], format='%Y-%m-%d').max()
        if df_start_date < min_start_date or df_end_date > max_end_date:
            warnings.warn(f'Stock {stock} has data out of the specified time range')
        # order the data by date from the earliest to the latest
        df = df.sort_values(by='date', ascending=True)
        df.to_csv(f'{save_path}/data_{num_stocks}_{"sentiment" if sentiment else "no_sentiment"}/{stock}',
                  index=False, )


def get_stocks_with_time_range(path, min_start_date, max_end_date, num_stocks):
    stocks = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            data = pd.read_csv(os.path.join(path, file))
            df = pd.DataFrame(data)
            start_date = pd.to_datetime(df['date'], format='%Y-%m-%d').min()
            end_date = pd.to_datetime(df['date'], format='%Y-%m-%d').max()
            if start_date <= min_start_date and end_date >= max_end_date:
                stocks.append(file)
            if len(stocks) == num_stocks:
                break
            if file == os.listdir(path)[-1]:
                warnings.warn(f'Specified range of time has not enough data for {num_stocks} stocks')
    return stocks


def adjust_features_names(path):
    for file in os.listdir(path):
        if file.endswith(".csv"):
            data = pd.read_csv(os.path.join(path, file))
            df = pd.DataFrame(data)
            df.rename(columns={'Date': 'date',
                               'Open': 'open',
                               'High': 'high',
                               'Low': 'low',
                               'Close': 'close',
                               'Adj close': 'adj_close',
                               'adj close': 'adj_close',
                               'Volume': 'volume',
                               'Sentiment_gpt': 'sentiment_gpt',
                               'News_flag': 'news_flag',
                               'Scaled_sentiment': 'scaled_sentiment', }, inplace=True),
            df.to_csv(os.path.join(path, file), index=False)


# Generate dataset with 30 stocks that have data in the time range 2013-01-01 to 2023-01-01
num_stocks = 50
generate_dataset(path='datasets/Stock_price/full_history', save_path='datasets',
                 min_start_date=pd.to_datetime('2013-1-1'), max_end_date=pd.to_datetime('2023-1-1'), num_stocks=num_stocks,
                 sentiment=False)
# Adjust the features names in the dataset
adjust_features_names(f'datasets/data_{num_stocks}_no_sentiment')
print("Feature names adjusted")
