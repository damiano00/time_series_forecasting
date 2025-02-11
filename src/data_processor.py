import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataPreprocessor(object):
    """ A class to load and preprocess the data """

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
        :param seq_len: length of the sequence
        :param normalise: whether to normalise the data
        :param cols_to_norm: columns to normalise
        :return: x, y, y_base
        """
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        y_base = data_windows[:, 0, [0]] #
        # data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows
        data_windows = self.normalise_selected_columns(data_windows, cols_to_norm, single_window=False) if normalise else data_windows
        cut_point = self.pred_len
        # x = data_windows[:, :-cut_point:]
        x = data_windows[:, :-1, :]
        y = data_windows[:, -1, [0]]
        return x,y,y_base

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
        """Generates the next data window from the given index location i"""
        window = self.data_train[i:i + seq_len]
        # window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        window = self.normalise_selected_columns(window, self.cols_to_norm, single_window=True)[
            0] if normalise else window
        # x = window[:-1]
        x = window[:-1]
        # y = window[0][2][0]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        """Normalise window with a base value of zero"""
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

    # Modified normalization function to normalize only specific columns
    def normalise_selected_columns(self, window_data, columns_to_normalise, single_window=False):
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

    def csv_to_df(path):
        """
        This function will load the CSV file from the given path
        :param path: path to the CSV file
        :return: Pandas DataFrame
        """
        try:
            df = pd.read_csv(path)
            return df
        except FileNotFoundError:
            print("The file was not found.")
            return None
        except pd.errors.EmptyDataError:
            print("No data found in the file.")
            return None
        except pd.errors.ParserError:
            print("Error parsing the file.")
            return None


    def get_stats(df):
        """
        This function will generate statistics for the given DataFrame and will return them as a dictionary, printing them
        :param df: DataFrame
        :return: dictionary containing statistics, or None if the DataFrame is empty
        """
        stats = {'shape': df.shape, 'columns': df.columns.tolist(), 'dtypes': df.dtypes.to_dict(),
                 'missing_values': df.isnull().sum().to_dict(), 'describe': df.describe().to_dict()}
        for key, value in stats.items():
            print(key, ":", value, '\n')
        # plot the timeseries
        plt.figure(figsize=(16, 8))
        df['date'] = pd.to_datetime(df['date'])
        plt.plot(df['date'], df['close'])
        plt.title('Close Price History')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price', fontsize=18)
        plt.show()
        return stats

    def reduce_dataset(path, output_path, no_lines_to_skip):
        """
        This function will reduce the dataset by the given percentage and save it to the output path
        :param path: path to the input CSV file
        :param output_path: path to the output CSV file
        :param no_lines_to_skip: number of lines to skip
        :return: None
        """
        # assert 1 <= no_lines_to_skip <= 10, "Percentage must be between 1 and 100."
        df = pd.read_csv(path, skiprows=lambda x: x % no_lines_to_skip != 0)
        df.to_csv(output_path, index=False)
        return df
