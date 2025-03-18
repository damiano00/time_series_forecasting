import re

import numpy as np
import os
import matplotlib.pyplot as plt


def main(n_stocks):
    if not os.path.exists('plots_comparison'):
        os.makedirs('plots_comparison')

    # Set the directory to scan
    directory = "./" # Current

    # Regex pattern to match folder names
    pattern = re.compile(r"^(sentiment|no_sentiment)_(\d+)_\d+")

    # List to store matching folder names
    matching_folders = []

    # Iterate over directory contents
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            match = pattern.match(item)
            if match:
                matching_folders.append(item)

    # Save the matching folders that contains the number of stocks
    sent_folders = [folder for folder in matching_folders if re.match(rf'sentiment_{n_stocks}_\d+', folder)][0]
    no_sent_folders = [folder for folder in matching_folders if re.match(rf'no_sentiment_{n_stocks}_\d+', folder)][0]


    # Load stored price data, ignoring date and time in folder name
    true_prices = np.load(f'{sent_folders}/evaluations/true_prices.npy')
    predicted_prices_sen = np.load(f'{sent_folders}/evaluations/predicted_prices.npy')
    predicted_prices_non_sen = np.load(f'{no_sent_folders}/evaluations/predicted_prices.npy')

    stock_idx = 1

    # Plot stock forecasting
    plt.figure(figsize=(12, 6))
    plt.plot(true_prices[:, stock_idx], label="True Future Prices", color='green', linewidth=2)
    plt.plot(predicted_prices_sen[:, stock_idx], label="Predicted using sentiment", color='red', linewidth=2)
    plt.plot(predicted_prices_non_sen[:, stock_idx], label="Predicted not using sentiment", color='blue', linewidth=2)

    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.title(f"Stock Forecasting for AAPL. Trained with {n_stocks} stocks")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots_comparison/stock_forecasting_AAPL_{n_stocks}.png')
    plt.show()

if __name__ == '__main__':
    main(5)
    main(25)
    main(48)