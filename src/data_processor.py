import numpy as np
import pandas as pd


class DataLoader:
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols, cols_to_norm, pred_len):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test = dataframe.get(cols).values[i_split:]
        self.cols_to_norm = cols_to_norm
        self.pred_len = pred_len
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise, cols_to_norm):
        """
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        """
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i + seq_len])
        data_windows = np.array(data_windows).astype(float)
        y_base = data_windows[:, 0, [0]]
        data_windows = self.normalise_selected_columns(data_windows, cols_to_norm,
                                                       single_window=False) if normalise else data_windows
        x = data_windows[:, :-1, :]
        y = data_windows[:, -1, [0]]
        return x, y, y_base

    def get_train_data(self, seq_len, normalise):
        """
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        """
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        """
        Yield a generator of training data from filename on given list of cols split for train/test
        @param seq_len: sequence length
        @param batch_size: batch size
        @param normalise: if True, normalise the data
        """
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        """
        Generates the next data window from the given index location i
        @param i: index location
        @param seq_len: sequence length
        @param normalise: if True, normalise the data
        """
        window = self.data_train[i:i + seq_len]
        window = self.normalise_selected_columns(window, self.cols_to_norm, single_window=True)[
            0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    @staticmethod
    def normalise_windows(window_data, single_window=False):
        """
        Normalise window with a base value of zero
        @param window_data: the window data
        @param single_window: if True, the window data is a single window
        """
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                w = window[0, col_i]
                if w == 0:
                    w = 1
                normalised_col = [((float(p) / float(w)) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(
                normalised_window).T  # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

    @staticmethod
    def normalise_selected_columns(window_data, columns_to_normalise, single_window=False):
        """
        Normalise only specific columns (Modified normalization function to normalise only specific columns)
        @param window_data: the window data
        @param columns_to_normalise: the columns to normalise
        @param single_window: if True, the window data is a single window
        """
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                if col_i in columns_to_normalise:
                    # Normalize only if the column index is in the list of columns to normalize
                    w = window[0, col_i]
                    if w == 0:
                        w = 1
                    normalised_col = [((float(p) / float(w)) - 1) for p in window[:, col_i]]
                else:
                    # Keep the original data for columns not in the list
                    normalised_col = window[:, col_i].tolist()
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
