import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


class Sensor:
    def __init__(self, sensor_id):
        self.sensor_id = sensor_id


overloaded_sensors = [Sensor(i) for i in range(5)]
underloaded_sensors = [Sensor(i+5) for i in range(5)]

# Calculate the radius for the circles
circle_radius = 1.5

# Set plot properties
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect("equal")
ax.set_xticks(np.arange(-6, 7))
ax.set_yticks(np.arange(-6, 7))
ax.set_xlabel("X")
ax.set_ylabel("Y")

# --- First circle (overloaded nodes) ---
circle_center_1 = (-3, 0)
circle_1 = plt.Circle(circle_center_1, circle_radius, color='gray', fill=False, linestyle='dashed', linewidth=2)
ax.add_artist(circle_1)

colors_over = cm.get_cmap('Reds', len(overloaded_sensors))

angles_1 = np.linspace(0, 2 * np.pi, len(overloaded_sensors), endpoint=False)
for idx, (angle, sensor) in enumerate(zip(angles_1, overloaded_sensors)):
    x_offset = circle_radius * np.cos(angle)
    y_offset = circle_radius * np.sin(angle)
    node_color = colors_over(idx)

    ax.scatter(circle_center_1[0] + x_offset, circle_center_1[1] + y_offset,
               color=node_color, marker='o', s=200, label=f"Overloaded Node {sensor.sensor_id + 1}")
    ax.text(circle_center_1[0] + x_offset, circle_center_1[1] + y_offset,
            str(sensor.sensor_id + 1), ha='center', va='center')

# --- Second circle (underloaded nodes) ---
circle_center_2 = (3, 0)
circle_2 = plt.Circle(circle_center_2, circle_radius, color='gray', fill=False, linestyle='dashed', linewidth=2)
ax.add_artist(circle_2)


colors_under = cm.get_cmap('Blues', len(underloaded_sensors))

angles_2 = np.linspace(0, 2 * np.pi, len(underloaded_sensors), endpoint=False)
for idx, (angle, sensor) in enumerate(zip(angles_2, underloaded_sensors)):
    x_offset = circle_radius * np.cos(angle)
    y_offset = circle_radius * np.sin(angle)
    node_color = colors_under(idx)

    ax.scatter(circle_center_2[0] + x_offset, circle_center_2[1] + y_offset,
               color=node_color, marker='o', s=200, label=f"Underloaded Node {sensor.sensor_id + 1}")
    ax.text(circle_center_2[0] + x_offset, circle_center_2[1] + y_offset,
            str(sensor.sensor_id + 1), ha='center', va='center')

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys())

plt.grid()
plt.tight_layout()
plt.title('Grouped Overloaded and Underloaded Nodes (Unique Colors)')
plt.savefig("Grouped_Overloaded_Underloaded_Nodes_UniqueColors.png", dpi=600, bbox_inches='tight')
plt.show()
