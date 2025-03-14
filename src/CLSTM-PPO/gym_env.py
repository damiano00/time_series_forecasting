import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import mahalanobis


class StockTradingEnv(gym.Env):
    """
    Custom Gym environment for automated multi-stock trading.
    The state is a 181-dimensional vector:
      - [0]: current available balance.
      - [1:31]: adjusted close prices for n stocks.
      - [31:61]: current holdings for n stocks.
      - [61:91]: MACD for n stocks.
      - [91:121]: RSI for n stocks.
      - [121:151]: CCI for n stocks.
      - [151:181]: ADX for n stocks.
      - [181:211]: Scaled sentiment for n stocks.
      - [211:241]: Sentiment GPT for n stocks.
      - [241:271]: News flag for n stocks.
    """

    def __init__(self, data, initial_balance=1e6, max_shares=100, reward_scaling=1e-4, turbulence_threshold=100,
                 state_dim=211, sentiment=True, n_stocks=30, scaler=MinMaxScaler()):
        super(StockTradingEnv, self).__init__()
        self.n_stocks = n_stocks
        self.data = data  # Data dictionary with keys: 'price', 'MACD', 'RSI', 'CCI', 'ADX'
        self.num_steps = self.data['price'].shape[0]
        self.initial_balance = initial_balance
        self.portfolio_history = []
        self.max_shares = max_shares
        self.reward_scaling = reward_scaling
        self.turbulence_threshold = turbulence_threshold
        self.state_dim = state_dim
        self.sentiment = sentiment
        self.scaler = scaler
        self.balance = initial_balance
        self.stock_owned = np.zeros(self.n_stocks, dtype=np.int32)
        self.current_step = 0
        self.stock_names = []
        self.past_returns = []
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
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
        if self.sentiment:
            scaled_sentiment = self.data['scaled_sentiment'][self.current_step, :].astype(np.float32)
            scaled_sentiment = np.nan_to_num(scaled_sentiment, nan=0.0)
            return macd, rsi, cci, adx, scaled_sentiment
        else:
            return macd, rsi, cci, adx

    def _get_state(self):
        if self.sentiment:
            macd, rsi, cci, adx, scaled_sentiment = self._get_indicators()
            state = np.concatenate((
                np.array([self.balance], dtype=np.float32),
                self.prices,
                self.stock_owned.astype(np.float32),
                macd, rsi, cci, adx, scaled_sentiment
            ))
            return state
        else:
            macd, rsi, cci, adx = self._get_indicators()
            state = np.concatenate((
                np.array([self.balance], dtype=np.float32),
                self.prices,
                self.stock_owned.astype(np.float32),
                macd, rsi, cci, adx
            ))
            return state

    def _get_portfolio_value(self):
        # Calculate portfolio value as balance plus value of holdings.
        return self.balance + np.dot(self.stock_owned, self.prices)

    def calculate_turbulence(self, current_returns):
        """
        Computes the turbulence index using the Mahalanobis distance.
        """
        if len(self.past_returns) < 30:  # Need at least 30 past days for a stable covariance matrix
            return 0  # No turbulence in the early phase

        # Convert past returns to NumPy array
        past_returns_array = np.array(self.past_returns[-30:])  # Last 30 days of data

        # Compute mean and covariance of historical returns
        mean_returns = np.mean(past_returns_array, axis=0)
        cov_matrix = np.cov(past_returns_array.T)

        # Regularization: Ensure covariance matrix is invertible
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

        # Compute Mahalanobis distance (turbulence)
        turbulence = mahalanobis(current_returns, mean_returns, np.linalg.inv(cov_matrix))
        return turbulence

    def step(self, agent_action):
        agent_action = np.clip(agent_action, -1, 1)
        trade_shares = np.nan_to_num(agent_action * self.max_shares, nan=0.0, posinf=self.max_shares,
                                     neginf=-self.max_shares).astype(np.int32)
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
        self.portfolio_history.append(current_value)  # Track portfolio value

        reward = (current_value - prev_value) * self.reward_scaling
        # penalize high volatility (Stronger penalty for excessive trading)
        reward = (current_value - prev_value) * self.reward_scaling
        reward -= 1e-3 * np.abs(np.mean(agent_action))  # Reduce weak trades
        reward += 0.01 * np.sign(current_value - prev_value)  # Encourage bigger profits
        turbulence = self.calculate_turbulence(self.past_returns)
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

    def get_metrics(self, predicted_data):
        """
        Computes various metrics for model evaluation.

        Args:
            predicted_data (np.array): The predicted portfolio values.

        Returns:
            dict: Dictionary containing MAE, MSE, R² score, MAPE, MedAE, and sMAPE.
        """
        # Compute actual portfolio values
        actual_values = np.array(self.portfolio_history)

        # Ensure predicted_data and actual_values have the same length
        min_len = min(len(predicted_data), len(actual_values))
        predicted_data = predicted_data[:min_len]
        actual_values = actual_values[:min_len]

        # Rescale back if prices were normalized using MinMaxScaler
        actual_values = self.scaler.inverse_transform(
            np.array(actual_values).reshape(-1, 1)
        ).flatten()

        predicted_data = self.scaler.inverse_transform(
            np.array(predicted_data).reshape(-1, 1)
        ).flatten()

        # Mean Absolute Percentage Error
        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        mape = mean_absolute_percentage_error(actual_values, predicted_data)

        # Median Absolute Error
        medae = median_absolute_error(actual_values, predicted_data)

        # Symmetric Mean Absolute Percentage Error
        def symmetric_mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

        smape = symmetric_mean_absolute_percentage_error(actual_values, predicted_data)

        return {
            'MAPE': mape,
            'MedAE': medae,
            'sMAPE': smape
        }

