from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model


def build_lstm_pre(time_window, state_dim, feature_dim=128):
    """
    LSTMpre: A feature extractor that takes a sequence of states (length=time_window, dimension=state_dim)
    and outputs a feature vector of dimension feature_dim.

    Architecture:
      - Input: A sequence of states (time_window x state_dim).
      - One LSTM layer that processes the input sequence and returns its final hidden state.
      - Three Dense (linear) layers, each followed by a Tanh activation.
      - Output: A feature vector of dimension feature_dim.

    Args:
      time_window (int): Number of time steps (length of the state sequence).
      state_dim (int): Dimensionality of each state vector.
      feature_dim (int): Dimensionality of the output feature vector (default is 128).

    Returns:
      model (tf.keras.Model): The LSTMpre feature extractor model.
    """
    inputs = Input(shape=(time_window, state_dim), name="state_sequence")

    # LSTM layer: processes the time-series sequence and returns the final hidden state.
    x = LSTM(units=feature_dim, activation='tanh', name="lstm_layer")(inputs)

    # Three linear (Dense) layers with Tanh activations.
    x = Dense(feature_dim, activation='tanh', name="dense_layer_1")(x)
    x = Dense(feature_dim, activation='tanh', name="dense_layer_2")(x)
    outputs = Dense(feature_dim, activation='tanh', name="dense_layer_3")(x)

    model = Model(inputs=inputs, outputs=outputs, name="LSTMpre")
    return model
