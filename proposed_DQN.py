import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class AllocationController:
    def __init__(self, max_resources=100):
        self.max_resources = max_resources

    def apply_scaling(self, action):
        if action == 0:
            adjustment = -10
        elif action == 1:
            adjustment = 0
        else:
            adjustment = 10
        return np.clip(adjustment, -self.max_resources, self.max_resources)


class CloudEnvironment:
    def __init__(self):
        self.current_resources = 50
        self.target_sla = 0.9
        self.total_requests = 0
        self.successful_tasks = 0
        self.downtime = 0

    def get_state(self):
        current_sla = np.clip(self.current_resources / 100, 0, 1)
        sla_deviation = self.target_sla - current_sla
        return np.array([self.current_resources / 100, sla_deviation]).reshape(1, -1)

    def step(self, adjustment):
        self.current_resources = np.clip(self.current_resources + adjustment, 0, 100)
        current_sla = np.clip(self.current_resources / 100, 0, 1)
        sla_deviation = abs(self.target_sla - current_sla)

        requests_this_step = random.randint(5, 10)
        self.total_requests += requests_this_step
        successful = int(requests_this_step * current_sla)
        self.successful_tasks += successful

        if current_sla < self.target_sla:
            self.downtime += 1

        reward = - (sla_deviation * 10 + self.current_resources * 0.1)
        done = False
        return self.get_state(), reward, done


def proposed(X_train, X_test, y_train, y_test):
    state_size = 2
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    env = CloudEnvironment()
    controller = AllocationController()
    print(X_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(y_train.shape)
    episodes = 1
    batch_size = 32
    rewards_per_episode = []

    # Tracking metrics
    resource_usage_history = []
    sla_history = []
    latency_history = []

    for e in range(episodes):
        state = env.get_state()
        total_reward = 0
        episode_start_time = time.time()

        for time_t in range(50):
            start_time = time.time()
            action = agent.act(state)
            adjustment = controller.apply_scaling(action)
            next_state, reward, done = env.step(adjustment)
            end_time = time.time()

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            resource_usage_history.append(env.current_resources)
            sla_history.append(env.current_resources / 100)
            latency_history.append((end_time - start_time) * 1000)

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        rewards_per_episode.append(total_reward)

        # Metrics
        actual_used = np.mean(resource_usage_history)
        allocated = controller.max_resources
        resource_utilization = (actual_used / allocated) * 100

        rmse = np.sqrt(mean_squared_error(sla_history, [env.target_sla] * len(sla_history)))
        actual_peak = np.max(sla_history)
        prediction_accuracy = (1 - (rmse / actual_peak)) * 100

        energy_consumed = np.sum([res * 0.01 for res in resource_usage_history])
        adaptation_latency = np.mean(latency_history)

        total_time_sec = time.time() - episode_start_time
        throughput = env.total_requests / total_time_sec
        task_completion_rate = (env.successful_tasks / env.total_requests) * 100
        availability = (1 - env.downtime / total_time_sec) * 100

        node_loads = np.random.randint(40, 60, size=5)  # Example node loads
        load_efficiency = 100 - (np.std(node_loads) / np.max(node_loads)) * 100

        over_provisioning = np.mean([(allocated - used) / allocated * 100 for used in resource_usage_history])
        total_demand = 70  # hypothetical demand
        unmet_demand = max(total_demand - np.mean(resource_usage_history), 0)
        under_provisioning = (unmet_demand / total_demand) * 100

        print(f"\nEpisode {e + 1}/{episodes}")
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



    # Plot reward over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_per_episode, marker='o', color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve: Reward per Episode")
    plt.grid(True)
    plt.show()
