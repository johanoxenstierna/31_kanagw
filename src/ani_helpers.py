
import cv2
import numpy as np
import random
from copy import deepcopy
import P
from scipy.stats import multivariate_normal
from src.trig_functions import min_max_normalization
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt

def decrement_all_index_axs0(index_removed, O0, waves=None):
	"""
	Whenever an axs0 is popped from the list, all index_axs0 with higher index will be wrong and
	need to be decremented by 1.
	For now it seems to only work with axs0
	"""

	for o0 in O0.values():
		if o0.index_axs0 != None:
			if o0.index_axs0 > index_removed:
				o0.index_axs0 -= 1
		for o1 in o0.O1.values():
			if o1.index_axs0 != None:
				if o1.index_axs0 > index_removed:
					o1.index_axs0 -= 1

			for o2_key, o2 in o1.O2.items():  # OBS THIS MEANS sps must have same or fewer frames than f
				if o2.index_axs0 != None:
					if o2.index_axs0 > index_removed:
						o2.index_axs0 -= 1

#
# PAINFUL 30 min BUG HERE (something messed up above)
#
# for sp_key, sp in sh.sps.items():
# 	if sp.index_axs0 != None and sp.o1 == None:
# 		if sp.index_axs0 > index_removed:
# 			sp.index_axs0 -= 1


def set_O1(o, ax_b, axs0):
	"""
	TODO: Test putting + transData in front  Prob not gonna go anything.
	AND add skewing
	Also, ax.plot() has a shadow transform:
	ax.plot(x, y, transform=shadow_transform, zorder=0.5*line.get_zorder())
	but it will add another ax to axs0
	"""

	'''TODO: object might need to be rotated around center instead of corner
	CHECK rotate_around, rotate TAKES ARGS'''
	M = mtransforms.Affine2D(). \
		scale(o.scale[o.clock]). \
		rotate(o.rotation[o.clock]). \
		translate(o.xy[o.clock][0], o.xy[o.clock][1]) + ax_b.transData

	o.ax0.set_alpha(o.alphas[o.clock])
	# o.ax0.set_zorder(deepcopy(int(o.zorder)))
	# o.ax0.set_zorder(1000)
	o.ax0.set_transform(M)

# xys_cur = [o.xy[o.clock, 0], o.xy[o.clock, 1]]
# axs0[o.index_axs0].set_data(xys_cur)  # SELECTS A SUBSET OF WHATS ALREADY PLOTTED
# plt.setp(axs0[o.index_axs0], markersize=10)
# axs0[o.index_axs0].set_alpha(o.alphas[o.clock])


# def set_O2(o2, axs0, axs1, ax_b, ii):
#
# 	# sp_len_cur = o2.sp_lens[o2.clock]
#
# 	# if o2.o0.id == 'projectiles':
# 	# 	if o2.clock < sp_len_cur + 1:  # beginning
# 	# 		xys_cur = [o2.xy[:o2.clock, 0], o2.xy[:o2.clock, 1]]  # list with 2 cols
# 	# 	else:
# 	# 		xys_cur = [o2.xy[o2.clock:o2.clock + sp_len_cur, 0], o2.xy[o2.clock:o2.clock + sp_len_cur, 1]]
# 	# elif o2.o0.id == 'waves':
# 	# 	if o2.clock < sp_len_cur + 1:  # beginning
# 	# 		xys_cur = [o2.xy[:o2.clock, 0], o2.xy[:o2.clock, 1]]  # list with 2 cols
# 	# 	else:
# 	xys_cur = [o2.xy[o2.clock, 0], o2.xy[o2.clock, 1]]
#
# 	# axs0[o2.index_axs0].set_data(xys_cur)  # SELECTS A SUBSET OF WHATS ALREADY PLOTTED
# 	axs0[o2.index_axs0].set_data(xys_cur)  # SELECTS A SUBSET OF WHATS ALREADY PLOTTED
# 	plt.setp(axs0[o2.index_axs0], markersize=10)
#
# 	if o2.o0.id == 'projectiles':
# 		axs0[o2.index_axs0].set_color((o2.R[o2.clock], o2.G[o2.clock], o2.B[o2.clock]))
# 	# axs0[sp.index_axs0].set_color('black')  # OBS
# 	# try:
# 	axs0[o2.index_axs0].set_alpha(o2.alphas[o2.clock])
# except:
# 	adf = 5




# def add_to_ars(sp, axs0):
#
# 	'''The main point of this is not to achieve any animation speed-up, which it wont, but rather
# 	to make things more sensical.'''
#
# 	'''Here add xy limit condition to see whether the arrow is actually relevant for ars'''
#
# 	aa = 5


