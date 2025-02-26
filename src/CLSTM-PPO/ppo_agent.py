import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import math
from lstm_pre import build_lstm_pre
from lstm_actor import build_lstm_actor
from lstm_critic import build_lstm_critic


# Assume the following functions are defined (from previous snippets):
# build_lstm_pre(time_window, state_dim, feature_dim)
# build_lstm_actor(feature_dim, n_stocks)
# build_lstm_critic(feature_dim)

class PPOAgent:
    def __init__(self, time_window, state_dim, feature_dim, n_stocks,
                 lr=3e-4, gamma=0.99, clip_epsilon=0.2, gae_lambda=0.95):
        self.time_window = time_window
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.n_stocks = n_stocks
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda

        # Build cascaded networks
        self.lstm_pre = build_lstm_pre(time_window, state_dim, feature_dim)
        self.actor = build_lstm_actor(feature_dim, n_stocks)
        self.critic = build_lstm_critic(feature_dim)

        # Log standard deviation as a trainable parameter for the Gaussian policy.
        self.log_std = tf.Variable(initial_value=tf.zeros(n_stocks), trainable=True, name="log_std")

        # Use a single optimizer for simplicity.
        self.optimizer = Adam(lr)
        self.mse_loss = MeanSquaredError()

        # Buffer for storing transitions
        self.buffer = []

    def select_action(self, state_seq):
        """
        Given a state sequence (shape: [time_window, state_dim]), this method:
          - Extracts features via lstm_pre.
          - Computes the action mean from the actor.
          - Samples an action from a diagonal Gaussian with the computed mean and a learned std.
          - Computes the log probability of the sampled action.
          - Returns the action (as a numpy array), its log probability, the critic value, and the feature vector.
        """
        # Add batch dimension: shape becomes (1, time_window, state_dim)
        state_seq = tf.expand_dims(state_seq, axis=0)
        features = self.lstm_pre(state_seq)  # shape: (1, feature_dim)
        mean = self.actor(features)  # shape: (1, n_stocks)

        # Compute standard deviation from log_std (broadcasted to batch size)
        std = tf.exp(self.log_std)
        std = tf.expand_dims(std, axis=0)  # shape: (1, n_stocks)

        # Sample action from Gaussian
        noise = tf.random.normal(shape=tf.shape(mean))
        action = mean + noise * std

        # Compute log probability of the sampled action
        var = std ** 2
        log_prob = -0.5 * tf.reduce_sum(((action - mean) ** 2) / var + 2 * tf.math.log(std) + tf.math.log(2 * math.pi),
                                        axis=1)

        # Get critic's value estimate
        value = self.critic(features)  # shape: (1, 1)
        return action[0].numpy(), log_prob[0].numpy(), value[0, 0].numpy()

    def store_transition(self, transition):
        self.buffer.append(transition)

    def clear_buffer(self):
        self.buffer = []

    def compute_returns_and_advantages(self, rewards, values, dones):
        """
        Computes returns and advantages using Generalized Advantage Estimation (GAE).
        """
        returns = []
        advantages = []
        gae = 0
        next_value = 0
        for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            if done:
                next_value = 0
                gae = 0
            delta = r + self.gamma * next_value - v
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v)
            next_value = v
        return returns, advantages

    def update(self, epochs=10, batch_size=64):
        # Convert stored transitions into tensors.
        states = tf.convert_to_tensor([t['state_seq'] for t in self.buffer],
                                      dtype=tf.float32)  # shape: (N, time_window, state_dim)
        actions = tf.convert_to_tensor([t['action'] for t in self.buffer], dtype=tf.float32)  # shape: (N, n_stocks)
        old_log_probs = tf.convert_to_tensor([t['log_prob'] for t in self.buffer], dtype=tf.float32)  # shape: (N,)
        returns = tf.convert_to_tensor([t['return'] for t in self.buffer], dtype=tf.float32)  # shape: (N,)
        advantages = tf.convert_to_tensor([t['advantage'] for t in self.buffer], dtype=tf.float32)  # shape: (N,)

        dataset = tf.data.Dataset.from_tensor_slices((states, actions, old_log_probs, returns, advantages))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

        for _ in range(epochs):
            for batch in dataset:
                s_batch, a_batch, old_logp_batch, ret_batch, adv_batch = batch
                with tf.GradientTape() as tape:
                    # Forward pass: extract features
                    features = self.lstm_pre(s_batch)  # shape: (batch, feature_dim)
                    mean = self.actor(features)  # shape: (batch, n_stocks)
                    std = tf.exp(self.log_std)
                    std = tf.expand_dims(std, axis=0)  # broadcast to (batch, n_stocks)
                    var = std ** 2

                    # New log probabilities
                    new_logp = -0.5 * tf.reduce_sum(
                        ((a_batch - mean) ** 2) / var + 2 * tf.math.log(std) + tf.math.log(2 * math.pi), axis=1)

                    ratio = tf.exp(new_logp - old_logp_batch)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * adv_batch, clipped_ratio * adv_batch))

                    # Critic loss (mean squared error)
                    value_preds = self.critic(features)
                    value_preds = tf.squeeze(value_preds, axis=1)
                    critic_loss = tf.reduce_mean(tf.square(ret_batch - value_preds))

                    # Entropy bonus for exploration
                    entropy = -tf.reduce_mean(new_logp)

                    total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                # Compute and apply gradients
                variables = self.lstm_pre.trainable_variables + self.actor.trainable_variables + self.critic.trainable_variables + [
                    self.log_std]
                grads = tape.gradient(total_loss, variables)
                self.optimizer.apply_gradients(zip(grads, variables))

        self.clear_buffer()

    def train(self, env, total_timesteps=10000, update_timestep=128):
        timestep = 0
        while timestep < total_timesteps:
            # For the cascaded architecture, the agent expects a sliding window of states.
            state = env.reset()  # Initial state from the environment (assumed shape: [state_dim])
            # Initialize state sequence as multiple copies of the initial state (for simplicity)
            state_seq = np.array([state] * self.time_window)
            done = False
            while not done:
                action, log_prob, value = self.select_action(state_seq)
                next_state, reward, done, _ = env.step(action)
                transition = {
                    'state_seq': state_seq,
                    'action': action,
                    'log_prob': log_prob,
                    'reward': reward,
                    'value': value,
                    'done': done
                }
                self.store_transition(transition)
                timestep += 1

                # Update state sequence with a sliding window: drop the oldest and append next_state.
                state_seq = np.vstack([state_seq[1:], next_state])

                if timestep % update_timestep == 0:
                    rewards = [t['reward'] for t in self.buffer]
                    values = [t['value'] for t in self.buffer]
                    dones = [t['done'] for t in self.buffer]
                    returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)
                    # Add computed returns and advantages to each transition.
                    for i in range(len(self.buffer)):
                        self.buffer[i]['return'] = returns[i]
                        self.buffer[i]['advantage'] = advantages[i]
                    self.update()
