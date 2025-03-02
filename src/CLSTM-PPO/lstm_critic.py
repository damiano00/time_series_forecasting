from tensorflow.keras.layers import Input, LSTM, Dense, Reshape
from tensorflow.keras.models import Model


def build_lstm_critic(feature_dim=128):
    """
    LSTM Critic Network for PPO.

    Architecture:
      - Input: A feature vector (from LSTMpre) with dimension `feature_dim`.
      - Reshape: Convert the feature vector into a sequence of length 1.
      - LSTM layer: Process the sequence with 128 hidden units.
      - Three Dense layers with Tanh activations.
      - Final Dense layer with linear activation to output a scalar value estimate.

    Args:
      feature_dim (int): Dimensionality of the input feature vector (default: 128).

    Returns:
      model (tf.keras.Model): The LSTM Critic model.
    """
    # Input from LSTMpre feature extractor
    inputs = Input(shape=(feature_dim,), name="critic_input")

    # Reshape to add a time dimension (sequence length = 1)
    x = Reshape((1, feature_dim), name="reshape_for_lstm")(inputs)

    # LSTM layer with 128 units, processing the sequence and outputting the final hidden state
    x = LSTM(units=128, activation='tanh', name="critic_lstm")(x)

    # Three dense layers with Tanh activation
    x = Dense(128, activation='tanh', name="critic_dense1")(x)
    x = Dense(128, activation='tanh', name="critic_dense2")(x)
    x = Dense(128, activation='tanh', name="critic_dense3")(x)

    # Final output layer: outputs a scalar state value
    outputs = Dense(1, activation='linear', name="critic_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="LSTMcritic")
    return model
