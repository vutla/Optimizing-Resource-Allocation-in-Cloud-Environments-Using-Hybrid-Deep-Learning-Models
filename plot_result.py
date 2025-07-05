import os
from save_load import *
os.makedirs('./Saved data/', exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

def plot_res():
    metrics = [
        "Resource Utilization",
        "Prediction Accuracy",
        "Energy Consumption",
        "Adaptation Latency",
        "Throughput",
        "Task Completion Rate",
        "Service Availability",
        "Load Balancing Efficiency",
        "Over-Provisioning Rate",
        "Under-Provisioning Rate"
    ]

    algorithms = ["DDQN", "LSTM+MCTS", "TADPG", "MADDPG", "PROPOSED"]

    data_70 = load('data_70')
    data_80 = load('data_80')

    # Plot each metric
    for i, metric in enumerate(metrics):
        values_70 = [data_70[j][i] for j in range(len(algorithms))]
        values_80 = [data_80[j][i] for j in range(len(algorithms))]

        x = np.arange(len(algorithms))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(x - width / 2, values_70, width, label='70%')
        plt.bar(x + width / 2, values_80, width, label='80%')

        plt.ylabel('Value')
        plt.xlabel('Algorithm')
        plt.title(f'{metric} Comparison')
        plt.xticks(x, algorithms)
        plt.legend(title='Training Split')
        plt.tight_layout()

        plt.savefig(f'./Result/{metric.replace(" ", "_")}_comparison.png')  # Save fig
        plt.show()


def validation_loss_graph():
    # Simulate 100 epochs
    epochs = np.arange(1, 101)
    proposed = load("PROPOSED")
    maddpg = load('MADDPG')
    tadpg = load("TADPG")
    lstm_mcts = load("LSTM+MCTS")
    ddqn = load("DDQN")
    # Smoothing for better readability
    ddqn = gaussian_filter1d(ddqn, sigma=1)
    lstm_mcts = gaussian_filter1d(lstm_mcts, sigma=1)
    tadpg = gaussian_filter1d(tadpg, sigma=1)
    maddpg = gaussian_filter1d(maddpg, sigma=1)
    proposed = gaussian_filter1d(proposed, sigma=1)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, ddqn, label='DDQN', color='blue', linewidth=2)
    plt.plot(epochs, lstm_mcts, label='LSTM+MCTS', color='orange', linewidth=2)
    plt.plot(epochs, tadpg, label='TADPG', color='green', linewidth=2)
    plt.plot(epochs, maddpg, label='MADDPG', color='red', linewidth=2)
    plt.plot(epochs, proposed, label='Proposed', color='purple', linewidth=2)

    # Axis labels
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)

    # Legend
    plt.legend(loc='upper right', fontsize=10)

    # Grid and styling
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Layout & save
    plt.tight_layout()
    plt.savefig("Result/validation_loss_graph.png", dpi=1000)
    plt.show()
