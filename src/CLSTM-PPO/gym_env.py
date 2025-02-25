import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class StockTradingEnv(gym.Env):
    """
    Custom Gym environment for automated multi-stock trading based on Zou et al. (2024).
    The state is a 181-dimensional vector:
      - [0]: current available balance.
      - [1:31]: adjusted close prices for 30 stocks.
      - [31:61]: current holdings (number of shares) for 30 stocks.
      - [61:91]: MACD for 30 stocks.
      - [91:121]: RSI for 30 stocks.
      - [121:151]: CCI for 30 stocks.
      - [151:181]: ADX for 30 stocks.

    The action is a vector in [-1,1] for each of the 30 stocks,
    representing normalized buy/sell signals (scaled by max_shares).
    The reward is the change in portfolio value (after transaction costs)
    scaled by a factor.
    """

    def __init__(self, df, initial_balance=1e6, max_shares=100, reward_scaling=1e-4, turbulence_threshold=100):
        super(StockTradingEnv, self).__init__()
        self.stock_owned = np.zeros(30, dtype=np.int32)
        self.current_step = 0
        self.balance = initial_balance
        self.df = df.reset_index(
            drop=True)  # DataFrame with at least 30 price columns; additional indicator columns optional.
        self.initial_balance = initial_balance
        self.max_shares = max_shares
        self.reward_scaling = reward_scaling
        self.turbulence_threshold = turbulence_threshold

        # Observation: 1 (balance) + 30 (prices) + 30 (holdings) + 4*30 (indicators) = 181 dims.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(181,), dtype=np.float32)

        # Action: 30-dimensional continuous vector, values in [-1,1].
        self.action_space = spaces.Box(low=-1, high=1, shape=(30,), dtype=np.float32)

        self.transaction_cost_pct = 0.001  # 0.1% per transaction

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = 0
        self.stock_owned = np.zeros(30, dtype=np.int32)
        self._update_current_prices()
        return self._get_state()

    def _update_current_prices(self):
        # Assume first 30 columns of df are adjusted close prices.
        self.prices = self.df.iloc[self.current_step, :30].values.astype(np.float32)

    @staticmethod
    def _get_indicators():
        # In practice, indicators like MACD, RSI, CCI, ADX should be computed.
        # Here we use placeholders (zeros) for simplicity.
        macd = np.zeros(30, dtype=np.float32)
        rsi = np.zeros(30, dtype=np.float32)
        cci = np.zeros(30, dtype=np.float32)
        adx = np.zeros(30, dtype=np.float32)
        return macd, rsi, cci, adx

    def _get_state(self):
        macd, rsi, cci, adx = self._get_indicators()
        state = np.concatenate((
            np.array([self.balance], dtype=np.float32),
            self.prices,
            self.stock_owned.astype(np.float32),
            macd, rsi, cci, adx
        ))
        return state

    def _get_portfolio_value(self):
        # Portfolio value = balance + sum(owned shares * current prices)
        return self.balance + np.dot(self.stock_owned, self.prices)

    def step(self, agent_action):
        # Clip action values and scale to number of shares.
        agent_action = np.clip(agent_action, -1, 1)
        trade_shares = (agent_action * self.max_shares).astype(np.int32)

        # Process trades for each stock.
        for i in range(30):
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

        # Save portfolio value before updating the step.
        prev_portfolio_value = self._get_portfolio_value()

        # Advance to the next time step.
        self.current_step += 1
        is_done = self.current_step >= len(self.df) - 1

        self._update_current_prices()
        current_portfolio_value = self._get_portfolio_value()

        # Reward: change in portfolio value scaled.
        reward = (current_portfolio_value - prev_portfolio_value) * self.reward_scaling

        # Optionally, check turbulence index (not computed here).
        turbulence = 0  # Placeholder; implement proper turbulence if data available.
        if turbulence > self.turbulence_threshold:
            is_done = True

        state = self._get_state()
        return state, reward, is_done, {}

    def render(self, mode='human'):
        portfolio_value = self._get_portfolio_value()
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Stock Owned: {self.stock_owned}")
        print(f"Portfolio Value: {portfolio_value:.2f}")


# Example usage:
if __name__ == '__main__':
    # For demonstration, create dummy data:
    # 500 time steps; 30 columns for prices (and additional columns if needed).
    dummy_data = pd.DataFrame(np.random.uniform(low=10, high=200, size=(500, 30)))

    env = StockTradingEnv(dummy_data)
    state = env.reset()
    done = False
    while not done:
        # Random action for demonstration.
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(agent_action=action)
        done = terminated or truncated
        env.render()
