import numpy as np
import os
import matplotlib.pyplot as plt

if not os.path.exists('plots_comparison'):
    os.makedirs('plots_comparison')

# Load stored price data
true_prices = np.load('sentiment_48_202503110339/evaluations/true_prices.npy')
predicted_prices_sen = np.load('sentiment_48_202503110339/evaluations/predicted_prices.npy')
predicted_prices_non_sen = np.load('no_sentiment_48_202503110424/evaluations/predicted_prices.npy')

stock_idx = 0

# Plot stock forecasting
plt.figure(figsize=(12, 6))
plt.plot(true_prices[:, stock_idx], label="True Future Prices", color='green', linewidth=2)
plt.plot(predicted_prices_sen[:, stock_idx], label="Predicted using sentiment", color='red', linewidth=2)
plt.plot(predicted_prices_non_sen[:, stock_idx], label="Predicted not using sentiment", color='blue', linewidth=2)

plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.title(f"Stock Forecasting for AAL")
plt.legend()
plt.grid(True)
plt.savefig('plots_comparison/stock_forecasting_AAL_48.png')
plt.show()
