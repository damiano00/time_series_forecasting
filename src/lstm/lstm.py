import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    This class defines the LSTM-based time series forecasting model.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initialize the LSTM model.
        :param input_size: The number of features in the input
        :param hidden_size: The number of hidden units in each LSTM layer
        :param num_layers: The number of LSTM layers
        :param output_size: The number of output units (number of labels)
        :param dropout: Dropout rate for regularization
        """
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the LSTM model.
        :param x: Input tensor of shape (batch_size, seq_len, input_size)
        :return: Output tensor of shape (batch_size, output_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM output
        out, _ = self.lstm(x, (h0, c0))

        # Get the output of the last time step
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)

        return out
