
import os
from scipy.stats._multivariate import multivariate_normal

import matplotlib.pyplot as plt
from src.trig_functions import _multivariate_normal

import numpy as np

PATH_IN = './pictures/waves/O1/'
PATH_OUT = './pictures/waves/O1/'
_, _, file_names = os.walk(PATH_IN).__next__()

rv = multivariate_normal(mean=[9, 9], cov=[[60, 0], [0, 60]])
x, y = np.mgrid[0:20:1, 0:20:1]
pos = np.dstack((x, y))
mask = rv.pdf(pos)
mask = mask / np.max(mask)


for file_name in file_names:
	pic = plt.imread(PATH_IN + file_name)
	alpha = pic[:, :, 3] * mask
	pic[:, :, 3] = alpha
	plt.imsave(PATH_OUT + file_name, pic)

	af = 5

# x, y = np.mgrid[-1:1:.01, -1:1:.01]
# x, y = np.mgrid[0:20:1, 0:20:1]
#
# pos = np.dstack((x, y))
#
# rv = multivariate_normal(mean=[9, 9], cov=[[20, 0], [0, 20]])
# # rv = multivariate_normal(mean=0.5, cov=1)
#
# fig0, ax0 = plt.subplots()
# #
# # ax2 = fig2.add_subplot(111)
#
# ax0.contourf(x, y, rv.pdf(pos))
#
# plt.show()
