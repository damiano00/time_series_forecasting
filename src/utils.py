import datetime as dt
import torch
import os


class Timer:

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))


class Environment:
	# CUDA availability
	def check_cuda(self, use_gpu=False):
		print("PyTorch Version:", torch.__version__)
		print("CUDA Available:", torch.cuda.is_available())

		if torch.cuda.is_available():
			self.setup_cuda_environment(use_gpu)
		else:
			print("CUDA is not available. Falling back to CPU.")

	@staticmethod
	def setup_cuda_environment(use_gpu):
		if use_gpu:
			print("CUDA Version:", torch.version.cuda)
			print("GPU Name:", torch.cuda.get_device_name(0))
			torch.device("cuda")
			os.environ["CUDA_VISIBLE_DEVICES"] = "0"
			print("Selected primary: GPU")
		else:
			torch.device("cpu")
			os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
			print("Selected primary: CPU")
