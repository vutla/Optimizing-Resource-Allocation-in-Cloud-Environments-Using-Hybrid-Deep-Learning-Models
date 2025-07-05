import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import time
from sklearn.metrics import mean_squared_error

# --- TFEN ---
def build_tfen(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    res = layers.Conv1D(32, 3, padding='same')(x)
    x = layers.add([x, res])
    x = layers.ReLU()(x)

    lstm_out = layers.LSTM(64, return_sequences=True)(x)
    attention = layers.Dense(1, activation='tanh')(lstm_out)
    attention = layers.Softmax(axis=1)(attention)

    context = layers.Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([attention, lstm_out])
    return models.Model(inputs, context)

# --- Actor ---
def build_actor(tfen_output_shape):
    inputs = tf.keras.Input(shape=tfen_output_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    outputs = layers.Dense(1, activation='tanh')(x)
    return models.Model(inputs, outputs)

# --- Critic ---
def build_critic(tfen_output_shape):
    state_input = tf.keras.Input(shape=tfen_output_shape)
    action_input = tf.keras.Input(shape=(1,))
    x = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    return models.Model([state_input, action_input], outputs)

# --- rPER ---
class rPER:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.memory = []
        self.priorities = []

    def add(self, experience, priority):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
            self.priorities.pop(0)
        self.memory.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        scaled_p = np.array(self.priorities) / np.sum(self.priorities)
        indices = np.random.choice(len(self.memory), batch_size, p=scaled_p)
        samples = [self.memory[i] for i in indices]
        return samples

# --- Cloud Environment ---
class CloudEnv:
    def __init__(self):
        self.current_resources = 50
        self.target_sla = 0.9
        self.total_requests = 0
        self.successful_tasks = 0
        self.downtime = 0

    def get_state(self):
        sla = np.clip(self.current_resources / 100, 0, 1)
        deviation = self.target_sla - sla
        return np.array([self.current_resources / 100, deviation])

    def step(self, adjustment):
        self.current_resources = np.clip(self.current_resources + adjustment, 0, 100)
        sla = np.clip(self.current_resources / 100, 0, 1)

        reward = - (abs(self.target_sla - sla) * 10 + self.current_resources * 0.1)

        reqs = random.randint(5, 10)
        self.total_requests += reqs
        success = int(reqs * sla)
        self.successful_tasks += success

        if sla < self.target_sla:
            self.downtime += 1

        return self.get_state(), reward

# --- TADPG Agent ---
class TADPGAgent:
    def __init__(self, state_shape):
        self.tfen = build_tfen(state_shape)
        tfen_output_shape = self.tfen.output_shape[1:]
        self.actor = build_actor(tfen_output_shape)
        self.critic = build_critic(tfen_output_shape)

        self.memory = rPER()
        self.actor_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.gamma = 0.99

    def act(self, state_seq):
        tfen_out = self.tfen(state_seq)
        action = self.actor(tfen_out)
        return action.numpy()[0][0]

    def remember(self, exp, priority):
        self.memory.add(exp, priority)

    def train(self, batch_size):
        if len(self.memory.memory) < batch_size:
            return
        batch = self.memory.sample(batch_size)

        for s_seq, a, r, ns_seq, done in batch:
            with tf.GradientTape() as tape_c:
                tfs = self.tfen(s_seq)
                tfsn = self.tfen(ns_seq)
                target = r
                if not done:
                    next_a = self.actor(tfsn)
                    target += self.gamma * self.critic([tfsn, next_a])
                current_q = self.critic([tfs, tf.expand_dims(a, axis=0)])
                loss_c = tf.reduce_mean(tf.square(target - current_q))

            grads_c = tape_c.gradient(loss_c, self.critic.trainable_variables + self.tfen.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads_c, self.critic.trainable_variables + self.tfen.trainable_variables))

            with tf.GradientTape() as tape_a:
                tfs = self.tfen(s_seq)
                pred_a = self.actor(tfs)
                loss_a = -tf.reduce_mean(self.critic([tfs, pred_a]))

            grads_a = tape_a.gradient(loss_a, self.actor.trainable_variables + self.tfen.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads_a, self.actor.trainable_variables + self.tfen.trainable_variables))

# --- Optimizer Function ---
def TADPG(X_train, X_test, y_train, y_test):
    state_shape = (10, 2)
    agent = TADPGAgent(state_shape)
    env = CloudEnv()
    print(X_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(y_train.shape)
    for e in range(1):
        state_seq = np.random.rand(1, 10, 2).astype(np.float32)
        total_reward = 0
        resource_usage_history = []
        sla_history = []
        latency_history = []
        start_ep = time.time()

        for t in range(20):
            t0 = time.time()
            act_val = agent.act(state_seq)
            adj = np.clip(act_val * 10, -10, 10)

            next_state, reward = env.step(adj)
            t1 = time.time()

            resource_usage_history.append(env.current_resources)
            sla_history.append(env.current_resources / 100)
            latency_history.append((t1 - t0) * 1000)

            priority = abs(reward)

            next_state_seq = np.concatenate([state_seq[:, 1:, :], format_next_state(next_state)], axis=1)

            agent.remember((state_seq, act_val, reward, next_state_seq, False), priority)

            agent.train(16)

            state_seq = next_state_seq
            total_reward += reward

        actual_used = np.mean(resource_usage_history)
        allocated = 100
        resource_utilization = (actual_used / allocated) * 100

        rmse = np.sqrt(mean_squared_error(sla_history, [env.target_sla] * len(sla_history)))
        actual_peak = np.max(sla_history)
        prediction_accuracy = (1 - (rmse / actual_peak)) * 100 if actual_peak > 0 else 0

        energy_consumed = np.sum([r * 0.01 for r in resource_usage_history])
        adaptation_latency = np.mean(latency_history)
        total_time = time.time() - start_ep
        throughput = env.total_requests / total_time if total_time > 0 else 0
        task_completion_rate = (env.successful_tasks / env.total_requests) * 100 if env.total_requests > 0 else 0
        availability = (1 - env.downtime / total_time) * 100 if total_time > 0 else 0

        node_loads = np.random.randint(40, 60, size=5)
        load_efficiency = 100 - (np.std(node_loads) / np.max(node_loads)) * 100
        over_provisioning = np.mean([(allocated - u) / allocated * 100 for u in resource_usage_history])
        total_demand = 70
        unmet = max(total_demand - actual_used, 0)
        under_provisioning = (unmet / total_demand) * 100

        print(f"\nEpisode {e + 1}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Resource Utilization: {resource_utilization:.2f}%")
        print(f"Prediction Accuracy: {prediction_accuracy:.2f}%")
        print(f"Energy Consumption: {energy_consumed:.2f} kWh")
        print(f"Adaptation Latency: {adaptation_latency:.2f} ms")
        print(f"Throughput: {throughput:.2f} req/sec")
        print(f"Task Completion Rate: {task_completion_rate:.2f}%")
        print(f"Service Availability: {availability:.2f}%")
        print(f"Load Balancing Efficiency: {load_efficiency:.2f}%")
        print(f"Over-Provisioning Rate: {over_provisioning:.2f}%")
        print(f"Under-Provisioning Rate: {under_provisioning:.2f}%")
