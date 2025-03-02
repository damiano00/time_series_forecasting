from tensorflow.keras.layers import Input, LSTM, Dense, Reshape
from tensorflow.keras.models import Model


def build_lstm_actor(feature_dim=128, n_stocks=30):
    """
    LSTM Actor Network for PPO.

    Architecture:
      - Input: A feature vector (output of LSTMpre) of dimension `feature_dim`.
      - Reshape: Convert the feature vector to a sequence of length 1 to allow LSTM processing.
      - LSTM layer: Processes the input sequence and outputs a 128-dimensional hidden state.
      - Three Dense layers with Tanh activation to further transform the features.
      - Final Dense layer with Tanh activation to output a continuous action vector of dimension `n_stocks`
        (each value in the range [-1, 1], representing normalized buy/sell signals for each stock).

    Args:
      feature_dim (int): Dimensionality of the input feature vector (default 128).
      n_stocks (int): Number of stocks, i.e., the dimension of the action output (default 30).

    Returns:
      model (tf.keras.Model): The LSTM Actor network.
    """
    # Input feature vector from the LSTMpre feature extractor
    inputs = Input(shape=(feature_dim,), name="actor_input")

    # Reshape input to have a time dimension (sequence length = 1)
    x = Reshape((1, feature_dim), name="reshape_for_lstm")(inputs)

    # LSTM layer with 128 hidden units; returns the final hidden state
    x = LSTM(units=128, activation='tanh', name="actor_lstm")(x)

    # Three Dense layers with Tanh activation for further processing
    x = Dense(128, activation='tanh', name="actor_dense1")(x)
    x = Dense(128, activation='tanh', name="actor_dense2")(x)
    x = Dense(128, activation='tanh', name="actor_dense3")(x)

    # Final output layer: outputs continuous actions for each stock in [-1, 1]
    outputs = Dense(n_stocks, activation='tanh', name="actor_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="LSTMactor")
    return model
