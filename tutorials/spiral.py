
import numpy as np
import matplotlib.pyplot as plt
from src.trig_functions import min_max_normalization
from scipy.stats import beta, gamma

fig, ax = plt.subplots(nrows=1, ncols=1)

NUM = 200

xy = np.zeros(shape=(NUM, 2))
# thetas = np.linspace(0, 10*np.pi, num=50)
thetas = np.linspace(0.6 * np.pi, -2 * np.pi, num=NUM)
# add_x = np.linspace(1, 100, num=NUM)
# add_y = np.linspace(1, 100, num=NUM)

add_x = np.linspace(0, 600, num=NUM)
sub_y = np.linspace(20, 100, num=NUM)
radiuss = np.linspace(100, 0.01, num=NUM)
radiuss = beta.pdf(x=np.linspace(0, 1, NUM), a=2, b=5, loc=0)
radiuss = min_max_normalization(radiuss, y_range=[5, 30])

# for theta in np.linspace(0, 10*np.pi):
for i in range(NUM):
    theta = thetas[i]

    # r = o1.gi['frames_tot'] - i  # radius
    r = radiuss[i]

    xy[i, 0] = r * np.cos(theta) + add_x[i]
    xy[i, 1] = r * np.sin(theta) - sub_y[i]

shift = np.full(shape=(xy.shape), fill_value=[200, 300])
xy[:, :] += shift


ax0 = plt.plot(xy[:, 0], xy[:, 1])
ax1 = plt.plot(xy[0, 0], xy[0, 1], color='blue', marker='o', markersize=10)
plt.show()