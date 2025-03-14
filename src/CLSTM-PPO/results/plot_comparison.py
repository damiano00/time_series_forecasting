import numpy as np
import os
import matplotlib.pyplot as plt

if not os.path.exists('plots_comparison'):
    os.makedirs('plots_comparison')

# Load stored price data
true_prices = np.load('old1/no_sentiment_5_202503130111/evaluations/true_prices.npy')
predicted_prices_sen = np.load('old1/sentiment_5_202503130047/evaluations/predicted_prices.npy')
predicted_prices_non_sen = np.load('old1/no_sentiment_5_202503130111/evaluations/predicted_prices.npy')

stock_idx = 1

# Plot stock forecasting
plt.figure(figsize=(12, 6))
plt.plot(true_prices[:, stock_idx], label="True Future Prices", color='green', linewidth=2)
plt.plot(predicted_prices_sen[:, stock_idx], label="Predicted using sentiment", color='red', linewidth=2)
plt.plot(predicted_prices_non_sen[:, stock_idx], label="Predicted not using sentiment", color='blue', linewidth=2)

plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.title(f"Stock Forecasting for AAPL")
plt.legend()
plt.grid(True)
plt.savefig('plots_comparison/stock_forecasting_AAPL_5.png')
plt.show()
