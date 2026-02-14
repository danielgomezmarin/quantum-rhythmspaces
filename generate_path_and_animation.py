# This script generates a path in the Hilbert space and creates an animation of the path in the potential landscape. It uses the functions in qufunctions.py.

""" --- IMPORT LIBRARIES --- """

from qufunctions import map_to_01, compute_t_to_angles, compute_shot_result, bit_string_to_hilbert, hilbert_to_pixel, gaussian_potential, linear_potential, path2txt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np


""" --- COMPUTE PATH --- """

def compute_new_position(x, y, vx, vy, potential):
    # x, y are the current position and potential is the potential function

    # Compute the gradient of the potential. Use finite differences
    h = 1e-2
    dVdx = (potential(x + h, y) - potential(x - h, y)) / (2 * h)
    dVdy = (potential(x, y + h) - potential(x, y - h)) / (2 * h)

    # Compute t_x and t_y
    t_x = map_to_01(dVdx)
    t_y = map_to_01(dVdy)

    # Compute the angles theta, phi, alpha
    angles_x, angles_y = compute_t_to_angles(t_x, t_y)

    # Compute the shot result
    bit_string = compute_shot_result(angles_x, angles_y)

    # Compute the new position in the Hilbert space
    x_hilbert, y_hilbert = bit_string_to_hilbert(bit_string)


    # NON-INERTIAL DYNAMICS
    # x_pixel, y_pixel = hilbert_to_pixel(x_hilbert, y_hilbert)
    # return x + x_pixel, y + y_pixel, vx, vy

    # INERTIAL DYNAMICS
    # Compute the new force in pixel space
    fx, fy = hilbert_to_pixel(x_hilbert, y_hilbert)
    m = 10 # mass
    delta_t = 0.52 # time
    # Update the velocity
    vx = vx + fx * delta_t / m
    vy = vy + fy * delta_t / m
    # Update the position
    x = x + vx * delta_t
    y = y + vy * delta_t

    return x, y, vx, vy


new_path_linear = [(0, 0)]
new_path_linear_v = [(0, 0)]
new_path_gaussian = [(100, 100)]
new_path_gaussian_v = [(10, -10)]

for i in range(1, 100): # Compute the new position 100 times
    x, y, vx, vy = new_path_linear[-1][0], new_path_linear[-1][1], new_path_linear_v[-1][0], new_path_linear_v[-1][1]
    x2, y2, vx2, vy2 = new_path_gaussian[-1][0], new_path_gaussian[-1][1], new_path_gaussian_v[-1][0], new_path_gaussian_v[-1][1]
    x, y, vx, vy = compute_new_position(x, y, vx, vy, linear_potential)
    x2, y2, vx2, vy2 = compute_new_position(x2, y2, vx2, vy2, gaussian_potential)
    new_path_linear.append((x, y))
    new_path_linear_v.append((vx, vy))
    new_path_gaussian.append((x2, y2))
    new_path_gaussian_v.append((vx2, vy2))

# Store the path in a txt file
path2txt(new_path_linear, "new_path_linear.txt")
path2txt(new_path_gaussian, "new_path_gaussian.txt")

""" --- POTENTIAL PLOTS & ANIMATIONS --- """

# Generate x and y values
x = np.arange(-250, 250, 1)
y = np.arange(-250, 250, 1)
X, Y = np.meshgrid(x, y)

# Calculate potential values
linear_V = linear_potential(X, Y)
gaussian_V = gaussian_potential(X, Y)


# Parameters for the plot
mpl.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.color": "black",
    "ytick.color": "black",
    "legend.fontsize": 12
})

# Select which potential to animate (Uncomment the one you want)

# # ----- LINEAR POTENTIAL -----
# fig, ax = plt.subplots(figsize=(7, 6))

# contour = ax.contourf(X, Y, linear_V, cmap='gray', levels=50)
# cbar = fig.colorbar(contour, ax=ax, label='Potential $V(x, y)$')
# cbar.ax.tick_params(labelsize=12)

# ax.set_title('Linear Potential')
# ax.set_xlabel('x (pixels)')
# ax.set_ylabel('y (pixels)')
# ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

# # Path data
# path_x = [pos[0] for pos in new_path_linear]
# path_y = [pos[1] for pos in new_path_linear]

# # Path line
# line, = ax.plot([], [], 'o-', color='steelblue', linewidth=2)

# # Axis and layout
# ax.set_xlim(-250, 250)
# ax.set_ylim(-250, 250)
# plt.tight_layout()

# # Animation function
# def animate(i):
#     line.set_data(path_x[:i+1], path_y[:i+1])
#     return [line]

# ani = animation.FuncAnimation(fig, animate, frames=len(path_x), interval=100, blit=True, repeat=False)
# # Save animation as video (e.g., MP4)
# ani.save('linear_potential_path.mp4', writer='ffmpeg', fps=10, dpi=150)

# # Export final frame
# def save_last_frame():
#     line.set_data(path_x, path_y)
#     # Highlight first and last points with larger markers
#     ax.plot(path_x[0], path_y[0], marker='o', color='steelblue', markersize=5)
#     fig.savefig('final_frame.pdf', dpi=900)

# ani._func(len(path_x)-1)  # update to final frame
# save_last_frame()

# plt.show()

# ----- GAUSSIAN POTENTIAL -----

# Plot setup
fig, ax = plt.subplots(figsize=(7, 6))

contour = ax.contourf(X, Y, gaussian_V, cmap='gray', levels=50)
cbar = fig.colorbar(contour, ax=ax, label='Potential $V(x, y)$')
cbar.ax.tick_params(labelsize=12)
ax.set_title('Inertial Potential')
ax.set_xlabel('x (pixels)')
ax.set_ylabel('y (pixels)')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

# Path data
path_x = [pos[0] for pos in new_path_gaussian]
path_y = [pos[1] for pos in new_path_gaussian]
line, = ax.plot([], [], '-', color='steelblue', linewidth=2)

# Animation function
def animate(i):
    line.set_data(path_x[:i+1], path_y[:i+1])
    return [line]

ani = animation.FuncAnimation(fig, animate, frames=len(path_x), interval=100, blit=True, repeat=False)
# Save animation as video (e.g., MP4)
ani.save('gaussian_potential_path.mp4', writer='ffmpeg', fps=10, dpi=150)

# Export final frame
def save_last_frame():
    line.set_data(path_x, path_y)
    # Highlight first and last points with larger markers
    ax.plot(path_x[0], path_y[0], marker='o', color='steelblue', markersize=5)
    fig.savefig('final_frame.pdf', dpi=900)

ani._func(len(path_x)-1)  # update to final frame
save_last_frame()

plt.show()

plt.show()








