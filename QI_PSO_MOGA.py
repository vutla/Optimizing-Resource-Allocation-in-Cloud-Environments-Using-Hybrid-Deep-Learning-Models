import numpy as np
import time

from matplotlib import pyplot as plt

def QI_PSO_MOGA():
    def objective_SLA(x):
        return np.sum(x) * 0.1

    def objective_energy(x):
        return np.sum(x ** 2) * 0.05

    def objective_cost(x):
        return np.sum(np.abs(x)) * 0.2

    def calculate_fitness(resource_allocation):
        total_resources = sum(resource_allocation)
        utilized_resources = sum(task_resources for task_resources in resource_allocation if task_resources > 0)
        fitness = utilized_resources / total_resources if total_resources > 0 else 0.0
        return fitness

    class ReinforcementAgent:
        def __init__(self):
            self.q_table = {}

        def get_action(self, state):
            if np.random.rand() < 0.2:
                return "explore"
            else:
                return "exploit"

        def update(self, state, action, reward, next_state):
            pass

    def qmogs_rl(problem, epoch=100, pop_size=30, alpha_max=1.0, alpha_min=0.3, crossover_prob=0.7, mutation_prob=0.1,
                 num_elite=2):
        lb = problem["lb"]
        ub = problem["ub"]
        fit_func = problem["fit_func"]
        minmax = problem["minmax"]
        D = len(lb)

        P = np.random.uniform(lb, ub, (pop_size, D))
        Pbest = P.copy()
        agent = ReinforcementAgent()
        task_execution_time = []

        fitness = np.array([fit_func(x) for x in P])
        Gbest_idx = np.argmin(fitness) if minmax == "min" else np.argmax(fitness)
        Gbest = P[Gbest_idx]
        Gbest_fit = fitness[Gbest_idx]

        task_start_time = time.time()

        for t in range(epoch):
            alpha = alpha_max - (alpha_max - alpha_min) * (t / epoch)
            mbest = np.mean(Pbest, axis=0)

            state = (t, Gbest_fit)
            action = agent.get_action(state)

            # Update positions
            for i in range(pop_size):
                phi = np.random.rand(D)
                p_local = phi * Pbest[i] + (1 - phi) * Gbest
                u = np.random.rand(D)
                if action == "explore":
                    P[i] = p_local + alpha * np.abs(P[i] - mbest) * np.log(1 / u)
                else:
                    P[i] = p_local - alpha * np.abs(P[i] - mbest) * np.log(1 / u)
                P[i] = np.clip(P[i], lb, ub)

            # Crossover
            offspring = []
            parents_idx = np.random.choice(pop_size, (pop_size // 2, 2))
            for p1, p2 in parents_idx:
                if np.random.rand() < crossover_prob:
                    point = np.random.randint(1, D)
                    c1 = np.concatenate([P[p1, :point], P[p2, point:]])
                    c2 = np.concatenate([P[p2, :point], P[p1, point:]])
                else:
                    c1, c2 = P[p1], P[p2]
                offspring.extend([c1, c2])
            P = np.array(offspring[:pop_size])

            # Mutation
            for i in range(pop_size):
                if np.random.rand() < mutation_prob:
                    dim = np.random.randint(D)
                    P[i, dim] = np.random.uniform(lb[dim], ub[dim])

            # Evaluate
            fitness = np.array([fit_func(x) for x in P])

            # Elitism
            elite_idx = np.argsort(fitness)[:num_elite] if minmax == "min" else np.argsort(fitness)[-num_elite:]
            P[:num_elite] = Pbest[elite_idx]

            # Update Pbest
            for i in range(pop_size):
                if (minmax == "min" and fitness[i] < fit_func(Pbest[i])) or (
                        minmax == "max" and fitness[i] > fit_func(Pbest[i])):
                    Pbest[i] = P[i]

            # Update Gbest
            Gbest_idx = np.argmin(fitness) if minmax == "min" else np.argmax(fitness)
            if (minmax == "min" and fitness[Gbest_idx] < Gbest_fit) or (
                    minmax == "max" and fitness[Gbest_idx] > Gbest_fit):
                Gbest = P[Gbest_idx]
                Gbest_fit = fitness[Gbest_idx]

            next_state = (t + 1, Gbest_fit)
            reward = -Gbest_fit if minmax == "min" else Gbest_fit
            agent.update(state, action, reward, next_state)

            # Timing
            task_end_time = time.time()
            task_execution_time.append(task_end_time - task_start_time)

        return Gbest, Gbest_fit, task_execution_time

    # --- Run ---
    if __name__ == "__main__":
        problem_dict1 = {
            "fit_func": calculate_fitness,
            "lb": np.array([-10, -15, -4, -2, -8, -9, -3, -3, -10, -12, -5, -3, -3, -14, -5]),
            "ub": np.array([10, 15, 12, 8, 15, 14, 12, 2, 6, 14, 12, 3, 5, 12, 10]),
            "minmax": "min",
        }

        best_position, best_fitness, task_times = qmogs_rl(problem_dict1, epoch=50, pop_size=30)
        print("Solution:", best_position)
        print("Fitness:", best_fitness)
        print("Task execution times (s):", task_times)

    # --- Sensor Class ---
    class Sensor:
        def __init__(self, sensor_id, workload):
            self.sensor_id = sensor_id
            self.workload = workload
            self.scaled_vm_workload = 0

        def scaling_work(self, scaled_workload):
            self.scaled_vm_workload = scaled_workload

    # --- Main ---
    if __name__ == "__main__":
        num_sensors = 25
        sensor_workloads = np.random.randint(1, 10, num_sensors)
        sensors = [Sensor(i, workload) for i, workload in enumerate(sensor_workloads)]

        for sensor in sensors:
            print(f"Sensor ID: {sensor.sensor_id}, Workload: {sensor.workload}")
        original_total_workload = sum(sensor.workload for sensor in sensors)
        print("Total Work Load for All Nodes:", original_total_workload)

        # Problem dict
        problem_dict = {
            "fit_func": calculate_fitness,
            "lb": np.zeros(num_sensors),
            "ub": np.ones(num_sensors) * 10,
            "minmax": "min",
        }

        # Run QMOGS-RL
        best_position, best_fitness, task_times = qmogs_rl(problem_dict, epoch=50, pop_size=30)
        print("Best Position (QMOGS):", best_position)
        print("Best Fitness (QMOGS):", best_fitness)

        total_resources = sum(best_position)

        for i, sensor in enumerate(sensors):
            allocated_resources = best_position[i]
            work = int(abs(allocated_resources / total_resources) * 10)
            sensor.scaling_work(work)

        for sensor in sensors:
            print(f"Sensor {sensor.sensor_id}: Scaled Workload {sensor.scaled_vm_workload}")

        scaled_vm_workload = sum(sensor.scaled_vm_workload for sensor in sensors)
        print("After Scaling Total Workload:", scaled_vm_workload)

        # Plot
        original_workloads = [sensor.workload for sensor in sensors]
        scaled_vm_workloads = [sensor.scaled_vm_workload for sensor in sensors]
        sensor_ids = [sensor.sensor_id for sensor in sensors]

        plt.figure(figsize=(10, 6))
        plt.plot(sensor_ids, original_workloads, marker='o', linestyle='-', color='blue', label='Original Workload')
        plt.plot(sensor_ids, scaled_vm_workloads, marker='s', linestyle='--', color='orange', label='Scaled Workload')
        plt.xlabel('Sensor ID')
        plt.ylabel('Workload')
        plt.title('Original vs Scaled Workload per Sensor (QMOGS-RL)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Optional: plot task execution time
        plt.figure(figsize=(10, 5))
        plt.plot(task_times, marker='.', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Cumulative Task Execution Time (s)')
        plt.title('Task Execution Time per Iteration')
        plt.grid(True)
        plt.show()

    import random

    class Sensor:
        def __init__(self, sensor_id, workload, scaled_vm_workload):
            self.sensor_id = sensor_id
            self.workload = workload
            self.scaled_vm_workload = scaled_vm_workload
            self.modified_workload = workload

        def allocate_work(self, extra_workload):
            self.modified_workload += extra_workload

    # Assuming vm_sensors is a list of Sensor objects
    vm_sensors = []

    # Total number of sensor nodes
    total_sensors = 25

    # Generate random values for workload and scaled workload for each sensor
    for sensor_id in range(total_sensors):
        workload = random.randint(1, 10)  # Random workload between 1 and 10
        scaled_vm_workload = random.randint(0, 10)  # Random scaled workload between 0 and 10
        vm_sensors.append(Sensor(sensor_id, workload, scaled_vm_workload))

    # Calculate priority based on modified workloads
    priority_dict = {sensor.sensor_id: sensor.scaled_vm_workload for sensor in vm_sensors}

    # Sort sensor nodes based on priority (higher modified workload means higher priority)
    sorted_sensors = sorted(vm_sensors, key=lambda sensor: priority_dict.get(sensor.sensor_id, 0), reverse=True)
    extra_workload = 100
    max_workload_per_sensor = 25

    scaled_vm_workload = sum(sensor.scaled_vm_workload for sensor in vm_sensors)

    # Allocate extra workload based on priority
    for sensor in sorted_sensors:
        # Calculate the maximum extra workload to be allocated to this sensor
        max_extra_workload_sensor = min(max_workload_per_sensor - sensor.scaled_vm_workload, extra_workload)
        # Update the workload for this sensor node
        sensor.allocate_work(max_extra_workload_sensor)

        # Update the remaining extra workload
        extra_workload -= max_extra_workload_sensor

    # Print sensor information
    for sensor in vm_sensors:
        print(
            f"Sensor ID: {sensor.sensor_id}, Workload: {sensor.workload}, Scaled Workload: {sensor.scaled_vm_workload}, Priority: {priority_dict.get(sensor.sensor_id, 0)}, Remaining Workload: {sensor.modified_workload}")



