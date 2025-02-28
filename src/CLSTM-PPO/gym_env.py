import gymnasium as gym
import numpy as np
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    """
    Custom Gym environment for automated multi-stock trading.
    The state is a 181-dimensional vector:
      - [0]: current available balance.
      - [1:31]: adjusted close prices for 30 stocks.
      - [31:61]: current holdings for 30 stocks.
      - [61:91]: MACD for 30 stocks.
      - [91:121]: RSI for 30 stocks.
      - [121:151]: CCI for 30 stocks.
      - [151:181]: ADX for 30 stocks.
      - [181:211]: Scaled sentiment for 30 stocks.
    """
    def __init__(self, data, initial_balance=1e6, max_shares=100, reward_scaling=1e-4, turbulence_threshold=100):
        super(StockTradingEnv, self).__init__()
        self.n_stocks = 30
        self.data = data  # Data dictionary with keys: 'price', 'MACD', 'RSI', 'CCI', 'ADX'
        self.num_steps = self.data['price'].shape[0]
        self.initial_balance = initial_balance
        self.max_shares = max_shares
        self.reward_scaling = reward_scaling
        self.turbulence_threshold = turbulence_threshold

        self.balance = initial_balance
        self.stock_owned = np.zeros(self.n_stocks, dtype=np.int32)
        self.current_step = 0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(211,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32)

        self.transaction_cost_pct = 0.001  # 0.1% transaction cost

        self._update_current_prices()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.stock_owned = np.zeros(self.n_stocks, dtype=np.int32)
        self.current_step = 0
        self._update_current_prices()
        return self._get_state()

    def _update_current_prices(self):
        # Update current prices from the 'price' array.
        self.prices = self.data['price'][self.current_step, :].astype(np.float32)

    def _get_indicators(self):
        # Get indicator values for the current time step.
        macd = self.data['MACD'][self.current_step, :].astype(np.float32)
        macd = np.nan_to_num(macd, nan=0.0)
        rsi = self.data['RSI'][self.current_step, :].astype(np.float32)
        rsi = np.nan_to_num(rsi, nan=0.0)
        cci = self.data['CCI'][self.current_step, :].astype(np.float32)
        cci = np.nan_to_num(cci, nan=0.0)
        adx = self.data['ADX'][self.current_step, :].astype(np.float32)
        adx = np.nan_to_num(adx, nan=0.0)
        sentiment_scaled = self.data['Scaled_sentiment'][self.current_step, :].astype(np.float32)
        sentiment_scaled = np.nan_to_num(sentiment_scaled, nan=0.0)
        return macd, rsi, cci, adx, sentiment_scaled

    def _get_state(self):
        macd, rsi, cci, adx, sentiment_scaled = self._get_indicators()
        state = np.concatenate((
            np.array([self.balance], dtype=np.float32),
            self.prices,
            self.stock_owned.astype(np.float32),
            macd, rsi, cci, adx, sentiment_scaled
        ))
        return state

    def _get_portfolio_value(self):
        # Calculate portfolio value as balance plus value of holdings.
        return self.balance + np.dot(self.stock_owned, self.prices)

    def step(self, agent_action):
        agent_action = np.clip(agent_action, -1, 1)
        # this gave enormous negative cumulative returns
        # trade_shares = (agent_action * self.max_shares).astype(np.int32)
        trade_shares = np.nan_to_num(agent_action * self.max_shares, nan=0.0, posinf=self.max_shares, neginf=-self.max_shares).astype(np.int32)

        for i in range(self.n_stocks):
            price = self.prices[i]
            if trade_shares[i] > 0:  # Buy action
                cost = trade_shares[i] * price * (1 + self.transaction_cost_pct)
                if cost <= self.balance:
                    self.balance -= cost
                    self.stock_owned[i] += trade_shares[i]
            elif trade_shares[i] < 0:  # Sell action
                sell_amount = min(int(abs(trade_shares[i])), int(self.stock_owned[i]))
                revenue = sell_amount * price * (1 - self.transaction_cost_pct)
                self.balance += revenue
                self.stock_owned[i] -= sell_amount

        prev_value = self._get_portfolio_value()
        self.current_step += 1
        done = self.current_step >= self.num_steps - 1
        self._update_current_prices()
        current_value = self._get_portfolio_value()

        reward = (current_value - prev_value) * self.reward_scaling
        turbulence = 0  # Placeholder for turbulence calculation
        if turbulence > self.turbulence_threshold:
            done = True

        state = self._get_state()
        return state, reward, done, {}

    def render(self, mode='human'):
        portfolio_value = self._get_portfolio_value()
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Stock Owned: {self.stock_owned}")
        print(f"Portfolio Value: {portfolio_value:.2f}")
