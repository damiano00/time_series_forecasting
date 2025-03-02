import os
import glob
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gym_env import StockTradingEnv  # Make sure your StockTradingEnv accepts a 'data' dict
from ppo_agent import PPOAgent
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(df, price_col='adj_close', span_short=12, span_long=26):
    ema_short = compute_ema(df[price_col], span=span_short)
    ema_long = compute_ema(df[price_col], span=span_long)
    macd = ema_short - ema_long
    return macd


def compute_rsi(df, price_col='adj_close', window=14):
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_cci(df, window=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad + 1e-8)
    return cci


def compute_adx(df, window=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    up_move = high - high.shift()
    down_move = low.shift() - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    plus_di = 100 * (plus_dm.rolling(window=window).sum() / (atr + 1e-8))
    minus_di = 100 * (minus_dm.rolling(window=window).sum() / (atr + 1e-8))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8))
    adx = dx.rolling(window=window).mean()
    return adx


def load_data(data_folder, n_stocks=30, sentiment=True):
    """
    Loads CSV files from the given folder, computes technical indicators (price, MACD, RSI, CCI, ADX)
    for each stock, aligns them by common dates, and returns a dictionary of NumPy arrays:
      - Each array has shape (num_timesteps, n_stocks)
    """
    check_stock_dates(data_folder)
    indicators = get_indicators(data_folder, sentiment=sentiment)
    common_index = determine_index_range(indicators)
    print(f"Common date range: {common_index[0]} to {common_index[-1]}")
    combined = {}
    for ind in indicators:
        df_list = []
        for key in sorted(indicators[ind].keys())[:n_stocks]:
            s = indicators[ind][key].reindex(common_index).ffill().bfill()
            df_list.append(s)
        combined[ind] = pd.concat(df_list, axis=1)
        combined[ind].columns = sorted(indicators[ind].keys())[:n_stocks]
    if len(common_index) == 0:
        raise ValueError("No common dates found among CSV files.")
    data_arrays = {ind: combined[ind].values for ind in combined}
    return data_arrays


def check_stock_dates(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        print("No CSV files found in the folder.")
        return
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        try:
            df = pd.read_csv(csv_file, parse_dates=['date'])
            if 'date' not in df.columns:
                raise ValueError(f"{file_name} does not contain a 'date' column.")
            if df.empty:
                raise ValueError(f"{file_name} is empty.")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")


def get_indicators(data_folder, sentiment):
    file_paths = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not file_paths:
        raise ValueError(f"No CSV files found in folder {data_folder}")
    if sentiment:
        indicators = {
            'price': {},
            'MACD': {},
            'RSI': {},
            'CCI': {},
            'ADX': {},
            'scaled_sentiment': {},
        }
    else:
        indicators = {
            'price': {},
            'MACD': {},
            'RSI': {},
            'CCI': {},
            'ADX': {},
        }
    for fp in file_paths:
        df = pd.read_csv(fp, parse_dates=['date'])
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        scaler = MinMaxScaler()
        df['adj_close'] = scaler.fit_transform(df['adj_close'].values.reshape(-1, 1)).flatten()
        price = df['adj_close']
        macd = compute_macd(df, price_col='adj_close')
        rsi = compute_rsi(df, price_col='adj_close')
        cci = compute_cci(df, window=20)
        adx = compute_adx(df, window=14)
        stock_key = os.path.splitext(os.path.basename(fp))[0]
        if sentiment:
            scaled_sentiment = np.mean(df['scaled_sentiment'].values)
            scaled_sentiment = pd.Series([scaled_sentiment] * len(price), index=price.index)
            indicators['scaled_sentiment'][stock_key] = scaled_sentiment
        indicators['price'][stock_key] = price
        indicators['MACD'][stock_key] = macd
        indicators['RSI'][stock_key] = rsi
        indicators['CCI'][stock_key] = cci
        indicators['ADX'][stock_key] = adx
    return indicators


def determine_index_range(indicators):
    price_data = indicators['price']
    stock_dates = {stock: (df.index.min(), df.index.max()) for stock, df in price_data.items()}
    highest_start_stock, (highest_start, _) = max(stock_dates.items(), key=lambda item: item[1][0])
    lowest_end_stock, (_, lowest_end) = min(stock_dates.items(), key=lambda item: item[1][1])
    common_start = max(start for start, _ in stock_dates.values())
    common_end = min(end for _, end in stock_dates.values())
    while common_start > common_end:
        price_data.pop(highest_start_stock, None)
        price_data.pop(lowest_end_stock, None)
        if not price_data:
            raise ValueError("No stocks remaining after exclusion. Check date ranges.")
        stock_dates = {stock: (df.index.min(), df.index.max()) for stock, df in price_data.items()}
        common_start = max(start for start, _ in stock_dates.values())
        common_end = min(end for _, end in stock_dates.values())
    print(f"Remaining stocks: {len(price_data)}")
    print(f"Common start date: {common_start}")
    print(f"Common end date: {common_end}")
    return pd.date_range(start=common_start, end=common_end, freq='D')


def backtest_agent(agent, env, time_window):
    state = env.reset()
    state_seq = np.array([state] * time_window)
    portfolio_values = []
    done = False
    initial_portfolio = env.balance + np.sum(env.prices * env.stock_owned)
    portfolio_values.append(initial_portfolio)
    while not done:
        action, _, _ = agent.select_action(state_seq)
        next_state, reward, done, _ = env.step(action)
        current_portfolio = env.balance + np.sum(env.prices * env.stock_owned)
        portfolio_values.append(current_portfolio)
        state_seq = np.vstack([state_seq[1:], next_state])
    return portfolio_values


def compute_performance_metrics(portfolio_values):
    portfolio_values = np.array(portfolio_values)
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    cumulative_return = (final_value - initial_value) / initial_value
    max_earning_rate = np.max((portfolio_values - initial_value) / initial_value)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)) * np.sqrt(252)
    return cumulative_return, max_earning_rate, sharpe_ratio


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow INFO & WARNING messages
    warnings.filterwarnings("ignore")  # Suppresses Python warnings
    tf.get_logger().setLevel('ERROR')  # Ensures TensorFlow's logger shows only errors

    curr_date = datetime.now().strftime("%Y%m%d%H%M%S")

    # Hyperparameters
    TIME_WINDOW = 30
    SENTIMENT = "no_sentiment"  # or "no_sentiment", this edit state dimension accordingly
    STATE_DIM = 1 + 30 * 7 if SENTIMENT == "sentiment" else 1 + 30 * 6 # 211 if sentiment, 181 if no_sentiment
    FEATURE_DIM = 128
    N_STOCKS = 30
    TOTAL_TIMESTEPS = 50000
    UPDATE_TIMESTEP = 128
    LR = 3e-4
    DATA_FOLDER = f"datasets/data_50"
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError(f"Data folder {DATA_FOLDER} does not exist.")

    # environment settings
    INITIAL_BALANCE = 1e6
    MAX_SHARES = 1000
    REWARD_SCALING = 1e-2
    TURBULENCE_THRESHOLD = 100

    # create folders for plots, saved models, and evaluations
    plots_path = f'results/{SENTIMENT}_{N_STOCKS}_{curr_date}/plots'
    saved_models_path = f'results/{SENTIMENT}_{N_STOCKS}_{curr_date}/saved_models'
    evaluations_path = f'results/{SENTIMENT}_{N_STOCKS}_{curr_date}/evaluations'
    if not os.path.exists(plots_path): os.makedirs(plots_path)
    if not os.path.exists(saved_models_path): os.makedirs(saved_models_path)
    if not os.path.exists(evaluations_path): os.makedirs(evaluations_path)

    # save the complete configuration in a file
    with open(f'results/{SENTIMENT}_{N_STOCKS}_{curr_date}/config.txt', 'w') as f:
        f.write(f'Time window: {TIME_WINDOW}\n'
                f'Sentiment: {SENTIMENT}\n'
                f'State dimension: {STATE_DIM}\n'
                f'Feature dimension: {FEATURE_DIM}\n'
                f'Number of stocks: {N_STOCKS}\n'
                f'Total timesteps: {TOTAL_TIMESTEPS}\n'
                f'Update timestep: {UPDATE_TIMESTEP}\n'
                f'Learning rate: {LR}\n'
                f'Initial balance: {INITIAL_BALANCE}\n'
                f'Max shares: {MAX_SHARES}\n'
                f'Reward scaling: {REWARD_SCALING}\n'
                f'Turbulence threshold: {TURBULENCE_THRESHOLD}\n')

    # print the configuration in the console through the file
    print(f'----- Training with {N_STOCKS} stocks and sentiment: {SENTIMENT} -----')
    with open(f'results/{SENTIMENT}_{N_STOCKS}_{curr_date}/config.txt', 'r') as f:
        print(f.read())

    # Load data and create environments
    data = load_data(data_folder=DATA_FOLDER, n_stocks=N_STOCKS, sentiment=(SENTIMENT == "sentiment"))
    total_steps = data['price'].shape[0]
    split_index = int(total_steps * 0.8)  # 80% in-sample, 20% out-of-sample

    train_data = {k: v[:split_index] for k, v in data.items()}
    test_data = {k: v[split_index:] for k, v in data.items()}

    train_env = StockTradingEnv(
        data=train_data,
        initial_balance=INITIAL_BALANCE,
        max_shares=MAX_SHARES,
        reward_scaling=REWARD_SCALING,
        turbulence_threshold=TURBULENCE_THRESHOLD,
        state_dim=STATE_DIM,
        n_stocks=N_STOCKS,
        sentiment=(SENTIMENT == "sentiment")
    )
    test_env = StockTradingEnv(
        data=test_data,
        initial_balance=INITIAL_BALANCE,
        max_shares=MAX_SHARES,
        reward_scaling=REWARD_SCALING,
        turbulence_threshold=TURBULENCE_THRESHOLD,
        state_dim=STATE_DIM,
        n_stocks=N_STOCKS,
        sentiment=(SENTIMENT == "sentiment")
    )
    agent = PPOAgent(
        time_window=TIME_WINDOW,
        state_dim=STATE_DIM,
        feature_dim=FEATURE_DIM,
        n_stocks=N_STOCKS,
        lr=LR
    )

    train_start_time = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f'----- Started training at: {train_start_time} -----')
    agent.train(train_env, total_timesteps=TOTAL_TIMESTEPS, update_timestep=UPDATE_TIMESTEP)
    train_end_time = datetime.now().strftime("%Y%m%d%H%M%S")
    # Convert the string times back to datetime objects
    start_dt = datetime.strptime(train_start_time, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(train_end_time, "%Y%m%d%H%M%S")
    # Calculate the difference in minutes
    minutes_difference = (end_dt - start_dt).total_seconds() / 60
    print(f'----- Finished training at: {train_end_time}, took {minutes_difference} minutes -----')


    agent.lstm_pre.save(os.path.join(saved_models_path, 'lstm_pre.keras'))
    agent.actor.save(os.path.join(saved_models_path, 'lstm_actor.keras'))
    agent.critic.save(os.path.join(saved_models_path, 'lstm_critic.keras'))

    print(f'evaluating agent...')
    portfolio_values = backtest_agent(agent, test_env, TIME_WINDOW)
    cr, mer, sr = compute_performance_metrics(portfolio_values)

    print(f"Cumulative Return: {cr * 100:.2f}%")
    print(f"Maximum Earning Rate: {mer * 100:.2f}%")
    print(f"Sharpe Ratio: {sr:.2f}")

    # save portfolio values to CSV
    pd.Series(portfolio_values).to_csv(os.path.join(evaluations_path, 'portfolio_values.csv'))
    # save cr, mer, sr to TXT
    with open(os.path.join(evaluations_path, f'performance_metrics_{SENTIMENT}'), "w") as f:
        f.write(f"Cumulative Return: {cr * 100:.2f}%\n")
        f.write(f"Maximum Earning Rate: {mer * 100:.2f}%\n")
        f.write(f"Sharpe Ratio: {sr:.2f}\n")

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label="Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'portfolio_value.png'))
    plt.show()

    # Plot equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label="Portfolio Value")
    plt.title("Equity Curve")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'equity_curve.png'))
    plt.show()

    # Compute drawdowns (percentage difference between a portfolio's value and its previous peak value)
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.cummax()
    drawdowns = (portfolio_series - rolling_max) / rolling_max

    plt.figure(figsize=(10, 6))
    plt.plot(drawdowns, label="Drawdown")
    plt.title("Portfolio Drawdown")
    plt.xlabel("Time Steps")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'portfolio_drawdown.png'))
    plt.show()

    # Compute daily returns and plot histogram
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    plt.figure(figsize=(10, 6))
    plt.hist(daily_returns, bins=30, edgecolor='k')
    plt.title("Histogram of Daily Returns")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'daily_returns_histogram.png'))
    plt.show()

    # Rolling volatility (e.g., 30-day volatility)
    rolling_vol = portfolio_series.pct_change().rolling(window=30).std() * np.sqrt(252)
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_vol, label="Rolling 30-day Volatility")
    plt.title("Rolling Volatility")
    plt.xlabel("Time Steps")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'rolling_volatility.png'))
    plt.show()

