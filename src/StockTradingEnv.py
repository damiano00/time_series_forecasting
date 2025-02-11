import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd


class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, datasets, window_size=30, initial_capital=1e6, trade_amount=1):
        super(StockTradingEnv, self).__init__()
        self.prev_portfolio_value = None
        self.holdings = None
        self.cash = None
        self.current_step = None
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.trade_amount = trade_amount

        # Ensure datasets is a list of DataFrames
        if isinstance(datasets, pd.DataFrame):
            datasets = [datasets]  # Wrap single DataFrame in a list
        elif not isinstance(datasets, list) or not all(isinstance(df, pd.DataFrame) for df in datasets):
            raise ValueError("datasets must be a DataFrame or a list of DataFrames")

        self.feature_cols = ['Open', 'High', 'Low', 'Close', 'Adj close',
                             'Volume', 'Sentiment_gpt', 'News_flag', 'Scaled_sentiment']
        self.feature_dim = len(self.feature_cols)
        self.n_stocks = len(datasets)

        # Convert dataframes to numpy arrays and sort by date
        self.stock_data = []
        for df in datasets:
            if 'Date' not in df.columns:
                raise ValueError("Each DataFrame must contain a 'Date' column")
            df = df.sort_values(by='Date')
            self.stock_data.append(df[self.feature_cols].to_numpy())

        # Ensure all stocks have the same number of timesteps
        self.T_total = min([data.shape[0] for data in self.stock_data])
        self.stock_data = [data[:self.T_total] for data in self.stock_data]
        self.data = np.stack(self.stock_data, axis=1)

        obs_dim = window_size * self.n_stocks * self.feature_dim + 1 + self.n_stocks
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.n_stocks, dtype=np.float32)
        self.prev_portfolio_value = self._get_portfolio_value()
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        action_mapped = np.array(action) - 1
        current_prices = self.data[self.current_step, :, 4]

        for i, act in enumerate(action_mapped):
            if act == 1 and self.cash >= current_prices[i] * self.trade_amount:
                self.cash -= current_prices[i] * self.trade_amount
                self.holdings[i] += self.trade_amount
            elif act == -1 and self.holdings[i] >= self.trade_amount:
                self.cash += current_prices[i] * self.trade_amount
                self.holdings[i] -= self.trade_amount

        current_portfolio_value = self._get_portfolio_value()
        reward = current_portfolio_value - self.prev_portfolio_value
        self.prev_portfolio_value = current_portfolio_value

        self.current_step += 1
        done = self.current_step >= self.T_total
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, False, {'portfolio_value': current_portfolio_value}

    def _get_portfolio_value(self):
        current_prices = self.data[self.current_step - 1, :, 4]
        return self.cash + np.sum(self.holdings * current_prices)

    def _get_observation(self):
        window_data = self.data[self.current_step - self.window_size:self.current_step].flatten()
        portfolio_info = np.concatenate(([self.cash], self.holdings))
        return np.concatenate((window_data, portfolio_info)).astype(np.float32)

    def render(self, mode='human'):
        print(
            f"Step: {self.current_step}, Portfolio Value: {self._get_portfolio_value():.2f}, Cash: {self.cash:.2f}, Holdings: {self.holdings}")
