import os
import glob
import numpy as np
import pandas as pd
from gym_env import StockTradingEnv  # Make sure your StockTradingEnv accepts a 'data' dict
from ppo_agent import PPOAgent
from sklearn.preprocessing import MinMaxScaler

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_macd(df, price_col='Adj close', span_short=12, span_long=26):
    ema_short = compute_ema(df[price_col], span=span_short)
    ema_long = compute_ema(df[price_col], span=span_long)
    macd = ema_short - ema_long
    return macd

def compute_rsi(df, price_col='Adj close', window=14):
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_cci(df, window=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad + 1e-8)
    return cci

def compute_adx(df, window=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
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

def load_data(data_folder="data", selected_stocks=30):
    """
    Loads CSV files from the given folder, computes technical indicators (price, MACD, RSI, CCI, ADX)
    for each stock, aligns them by common dates, and returns a dictionary of NumPy arrays:
      - Each array has shape (num_timesteps, n_stocks)
    """
    # Get indicators for each stock
    indicators = get_indicators(data_folder)
    # Determine common index range
    common_index = determine_index_range(indicators)
    print(f"Common date range: {common_index[0]} to {common_index[-1]}")
    # Reindex each series to the common_index and forward/backward fill missing values.
    combined = {}
    for ind in indicators:
        df_list = []
        for key in sorted(indicators[ind].keys())[:selected_stocks]:
            s = indicators[ind][key].reindex(common_index).ffill().bfill()
            df_list.append(s)
        combined[ind] = pd.concat(df_list, axis=1)
        combined[ind].columns = sorted(indicators[ind].keys())[:selected_stocks]
    if len(common_index) == 0:
        raise ValueError("No common dates found among CSV files.")
    data_arrays = {ind: combined[ind].values for ind in combined}
    return data_arrays

def get_indicators(data_folder):
    file_paths = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    if not file_paths:
        raise ValueError(f"No CSV files found in folder {data_folder}")

    indicators = {
        'price': {},
        'MACD': {},
        'RSI': {},
        'CCI': {},
        'ADX': {}
    }

    # Process each CSV file
    for fp in file_paths:
        df = pd.read_csv(fp, parse_dates=['Date'])
        # Convert to date (dropping time) if desired
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
        # scaling with MinMaxScaler
        scaler = MinMaxScaler()
        df['Adj close'] = scaler.fit_transform(df['Adj close'].values.reshape(-1, 1)).flatten()
        # Extract price data
        price = df['Adj close']
        # Compute indicators
        macd = compute_macd(df, price_col='Adj close')
        rsi = compute_rsi(df, price_col='Adj close')
        cci = compute_cci(df, window=20)
        adx = compute_adx(df, window=14)
        # Store indicators in the dictionary
        stock_key = os.path.splitext(os.path.basename(fp))[0]
        indicators['price'][stock_key] = price
        indicators['MACD'][stock_key] = macd
        indicators['RSI'][stock_key] = rsi
        indicators['CCI'][stock_key] = cci
        indicators['ADX'][stock_key] = adx
    return indicators


def determine_index_range(indicators):
    price_data = indicators['price']
    # Map each stock to its start and end dates.
    stock_dates = {
        stock: (df.index.min(), df.index.max())
        for stock, df in price_data.items()
    }
    # Identify stocks with the highest start and lowest end dates.
    highest_start_stock, (highest_start, _) = max(stock_dates.items(), key=lambda item: item[1][0])
    lowest_end_stock, (_, lowest_end) = min(stock_dates.items(), key=lambda item: item[1][1])
    common_start = max(start for start, _ in stock_dates.values())
    common_end = min(end for _, end in stock_dates.values())
    # If there's no overlap, exclude the outlier stocks and try again.
    while common_start > common_end:
        price_data.pop(highest_start_stock, None)
        price_data.pop(lowest_end_stock, None)
        if not price_data:
            raise ValueError("No stocks remaining after exclusion. Check date ranges.")
        stock_dates = {
            stock: (df.index.min(), df.index.max())
            for stock, df in price_data.items()
        }
        common_start = max(start for start, _ in stock_dates.values())
        common_end = min(end for _, end in stock_dates.values())
    print(f"Remaining stocks: {len(price_data)}")
    return pd.date_range(start=common_start, end=common_end, freq='D')


def backtest_agent(agent, env, time_window):
    """
    Runs the trained agent on the test environment and records portfolio values.
    """
    state = env.reset()
    # Initialize a state sequence by repeating the initial state
    state_seq = np.array([state] * time_window)
    portfolio_values = []
    done = False

    # Record initial portfolio value
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
    """
    Computes performance metrics:
      - Cumulative Return (CR)
      - Maximum Earning Rate (MER)
      - Sharpe Ratio (SR)
    """
    portfolio_values = np.array(portfolio_values)
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    cumulative_return = (final_value - initial_value) / initial_value
    max_earning_rate = np.max((portfolio_values - initial_value) / initial_value)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)) * np.sqrt(252)
    return cumulative_return, max_earning_rate, sharpe_ratio

if __name__ == "__main__":
    # Hyperparameters
    TIME_WINDOW = 30
    STATE_DIM = 1 + 30 * 6  # [balance, prices, holdings, MACD, RSI, CCI, ADX] = 181
    FEATURE_DIM = 128
    N_STOCKS = 30
    TOTAL_TIMESTEPS = 10000
    UPDATE_TIMESTEP = 128
    LR = 3e-4

    # Load data from CSV files in the "data" folder
    data = load_data(data_folder="data", selected_stocks=N_STOCKS)
    total_steps = data['price'].shape[0]
    split_index = int(total_steps * 0.8)  # 80% for in-sample training, 20% for out-of-sample testing

    # Create in-sample and out-of-sample data dictionaries
    train_data = {k: v[:split_index] for k, v in data.items()}
    test_data = {k: v[split_index:] for k, v in data.items()}

    # Create training and testing environments (StockTradingEnv expects a data dict)
    train_env = StockTradingEnv(
        data=train_data,
        initial_balance=1e6,
        max_shares=100,
        reward_scaling=1e-4,
        turbulence_threshold=100
    )
    test_env = StockTradingEnv(
        data=test_data,
        initial_balance=1e6,
        max_shares=100,
        reward_scaling=1e-4,
        turbulence_threshold=100
    )

    # Instantiate and train the PPO agent on in-sample data
    agent = PPOAgent(
        time_window=TIME_WINDOW,
        state_dim=STATE_DIM,
        feature_dim=FEATURE_DIM,
        n_stocks=N_STOCKS,
        lr=LR
    )
    agent.train(train_env, total_timesteps=TOTAL_TIMESTEPS, update_timestep=UPDATE_TIMESTEP)

    # Optionally, save trained models
    agent.lstm_pre.save("models/lstm_pre.keras")
    agent.actor.save("models/lstm_actor.keras")
    agent.critic.save("models/lstm_critic.keras")

    # Backtest the trained agent on out-of-sample data
    portfolio_values = backtest_agent(agent, test_env, TIME_WINDOW)
    cr, mer, sr = compute_performance_metrics(portfolio_values)

    print(f"Cumulative Return: {cr * 100:.2f}%")
    print(f"Maximum Earning Rate: {mer * 100:.2f}%")
    print(f"Sharpe Ratio: {sr:.2f}")
