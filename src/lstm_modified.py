import os
import numpy as np
import datetime as dt
from numpy import newaxis
from utils import Timer
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Model:
    """A class for building and inferencing a LSTM model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print(f"[Model] Loading model from file {filepath}")
        self.model = load_model(filepath)

    def build_model(self, configs):
        """
        Build the model with the given configuration
        :param configs: dictionary containing configuration for the model
        """
        timer = Timer()
        timer.start()
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
        print("[Model] Model Compiled")
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        """
        Train the model
        :param x: the input data
        :param y: the output data
        :param epochs: the number of epochs
        :param batch_size: the batch size
        :param save_dir: the save directory
        """
        timer = Timer()
        timer.start()
        print("[Model] Training Started")
        print(f"[Model] {epochs} epochs, {batch_size} batch size")
        save_filename = os.path.join(save_dir, f"{dt.datetime.now().strftime('%d%m%Y-%H%M%S')}-e{epochs}.keras")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_filename, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(save_filename)
        print(f"[Model] Training Completed. Model saved as {save_filename}")
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir, sentiment_type, model_name,
                        num_csvs):
        """
        Train the model using a generator
        :param data_gen: the data generator
        :param epochs: the number of epochs
        :param batch_size: the batch size
        :param steps_per_epoch: the number of steps per epoch
        :param save_dir: the save directory
        :param sentiment_type: the sentiment type
        :param model_name: the model name
        :param num_csvs: the number of csvs
        """
        timer = Timer()
        timer.start()
        print("[Model] Training Started")
        print(f"[Model] {epochs} epochs, {batch_size} batch size, {steps_per_epoch} batches per epoch")
        model_path = f"{model_name}_{sentiment_type}_{num_csvs}.h5"
        save_filename = os.path.join(save_dir, model_path)
        callbacks = [
            ModelCheckpoint(filepath=save_filename, monitor='loss', save_best_only=True)
        ]
        self.model.fit(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )
        print(f"[Model] Training Completed. Model saved as {save_filename}")
        timer.stop()

    def predict_point_by_point(self, data):
        """
        Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        :param data: the data to predict
        """
        print("[Model] Predicting point by point...")
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        """
        Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        :param data: the data to predict
        :param window_size: the window size
        :param prediction_len: the prediction length
        """
        print("[Model] Predicting Sequences Multiple...")
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        """
        Shift the window by 1 new prediction each time, re-run predictions on new window
        :param data: the data to predict
        :param window_size: the window size
        """
        print("[Model] Predicting Sequences Full...")
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted
