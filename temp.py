
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from src.trig_functions import *

num = 60

xy_tp = np.zeros(shape=(num, 2))

thetas = np.linspace(0.5 * np.pi, 0, num=num)  # for some reason, 0.5 pi is straight right, and 0 is straight down
# thetas = np.linspace(0.6 * np.pi, -1 * np.pi, num=o1f_f.gi['frames_tot'])\

x_end = num * 1.7
y_end = -60
shift_x = np.linspace(0, x_end, num=num)
shift_y = np.linspace(0, y_end, num=num)

radiuss = np.linspace(0, 50.1, num=num)
# radiuss = beta.pdf(x=np.linspace(0.99, 1, num), a=2, b=5, loc=0)
# radiuss = min_max_normalization(radiuss, y_range=[5, 50])

for i in range(num):
	theta = thetas[i]

	r = radiuss[i]  # displacement per frame
	# r = 1

	# xy_tp[i, 0] = shift_x[i]
	# xy_tp[i, 1] = shift_y[i]

	xy_tp[i, 0] = r * np.cos(theta) + shift_x[i]
	xy_tp[i, 1] = r * np.sin(theta) + shift_y[i]

	# y = gi['v'] * np.sin(gi['theta']) * t_lin - 0.5 * G * t_lin ** 2

v = 100.7
theta = 0
G = 9.8
h = 50
t_flight = (v * np.sin(theta) + np.sqrt((v * np.sin(theta))**2 + 2 * G * h)) / G
t_lin = np.linspace(0, t_flight, num)
xy_tp[:, 0] = v * np.cos(theta) * t_lin
xy_tp[:, 1] = v * np.sin(theta) * 2 * t_lin - 0.5 * G * t_lin ** 2

fig, ax = plt.subplots()
ax.plot(xy_tp[0, 0], xy_tp[0, 1], marker='o')
ax.plot(xy_tp[:, 0], xy_tp[:, 1])
plt.show()

asdf = 5