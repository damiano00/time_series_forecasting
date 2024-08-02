from tensorflow.keras import Sequential, Model, load_model
from src.utils import Timer
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
from src.data_processor import get_stats, reduce_dataset

class Model():
    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()




