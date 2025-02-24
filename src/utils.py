import os
import torch
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt


class Timer:

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))


class Environment:
	def __init__(self, use_gpu):
		self.use_gpu = use_gpu

	# CUDA availability
	def check_cuda(self):
		print("PyTorch Version:", torch.__version__)
		print("CUDA Available:", torch.cuda.is_available())
		if torch.cuda.is_available():
			self.setup_cuda_environment()
		else:
			print("CUDA is not available. Falling back to CPU.")


	def setup_cuda_environment(self):
		if self.use_gpu:
			print("CUDA Version:", torch.version.cuda)
			print("GPU Name:", torch.cuda.get_device_name(0))
			torch.device("cuda")
			os.environ["CUDA_VISIBLE_DEVICES"] = "0"
			print("Selected primary: GPU")
		else:
			torch.device("cpu")
			os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
			print("Selected primary: CPU")


class Plotter:


	def plot_one_feature_over_time(self, feature, stock_name, dir_path = '../dataset/processed/data_for_lstm'):
		"""
		Plot one feature over time.
        """
		stock_data = pd.read_csv(os.path.join(dir_path, stock_name+'.csv'))
		stock_data['Date'] = stock_data['Date'].astype(str)  # Ensure string format
		stock_data['Date'] = stock_data['Date'].apply(self.str_to_datetime)  # Convert to datetime.date
		stock_data.set_index('Date', inplace=True)
		plt.figure(figsize=(18, 9))
		plt.plot(range(stock_data.shape[0]), (stock_data[feature]))
		print(stock_data[feature])
		print(stock_data[feature].shape)
		step = max(stock_data.shape[0] // 25, 1)
		plt.xticks(range(0, stock_data.shape[0], step), stock_data.index[::step], rotation=45)
		plt.title(f"{stock_name} {feature}")
		plt.xlabel('Date', fontsize=18)
		plt.ylabel(feature, fontsize=18)
		plt.show()


	@staticmethod
	def plot_close_prices(dates, true_close, predicted_close):
		"""
		Plot two graphs for the true and predicted close prices over time.

		Parameters:
		- dates: An iterable of datetime objects or strings representing the dates.
		- true_close: Iterable of true close prices.
		- predicted_close: Iterable of predicted close prices.
		"""
		fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

		axs[0].plot(dates, true_close, color='blue', label='True Close Price')
		axs[0].set_title('True Close Price Over Time')
		axs[0].set_ylabel('Price')
		axs[0].legend()
		axs[0].grid(True)

		axs[1].plot(dates, predicted_close, color='red', label='Predicted Close Price')
		axs[1].set_title('Predicted Close Price Over Time')
		axs[1].set_xlabel('Date')
		axs[1].set_ylabel('Price')
		axs[1].legend()
		axs[1].grid(True)

		plt.tight_layout()
		plt.show()

	@staticmethod
	def str_to_datetime(timestamp: str):
		"""
		Convert string timestamp to datetime.date.
		"""
		try:
			return dt.datetime.fromisoformat(timestamp).date()
		except ValueError:
			raise ValueError(f"Invalid date format: {timestamp}")