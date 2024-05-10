import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define Gerstner wave parameters
amplitude = 0.1
frequency = 0.5
direction = np.array([1, 1])  # Direction vector (x, y)
phase_shift = np.pi / 2  # Phase shift

# Define grid parameters
x_min, x_max = -5, 5
y_min, y_max = -5, 5
num_points = 100
t_max = 10
num_frames = 100

# Generate grid
x = np.linspace(x_min, x_max, num_points)
y = np.linspace(y_min, y_max, num_points)
X, Y = np.meshgrid(x, y)

# Initialize figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize points
points, = ax.plot([], [], [], 'bo')

# Function to update animation
def update(frame):
    t = frame * t_max / num_frames
    Z = np.zeros_like(X)
    for i in range(num_points):
        for j in range(num_points):
            position = np.array([X[i, j], Y[i, j]])
            # Calculate Gerstner wave height at the current point
            displacement = amplitude * np.cos(2 * np.pi * frequency * (position.dot(direction)) + t - phase_shift)
            Z[i, j] = displacement
    points.set_data(X.flatten(), Y.flatten())
    points.set_3d_properties(Z.flatten())
    return points,

# Set axis limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(-amplitude, amplitude)

# Create animation
ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

plt.show()
