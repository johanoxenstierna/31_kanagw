
import numpy as np
import copy
import matplotlib.pyplot as plt

import P
from src.trig_functions import min_max_normalization
import random
import scipy
from scipy.stats import beta, gamma


def gerstner_waves(gi):

	"""
	Per particle!
	3 waves:
	0: The common ones
	1: The big one
	2: The small ones in opposite direction
	"""

	# lam = 1.5  # np.pi / 2 - 0.07  # pi is divided by this, WAVELENGTH
	# lam = 200  # np.pi / 2 - 0.07  # pi is divided by this, WAVELENGTH, VERY SENSITIVE

	# c = 0.5
	# c = -np.sqrt(9.8 / k)
	stn_particle = gi['steepness']

	# left_start = gi['o1_left_start']

	frames_tot = gi['frames_tot']

	d = np.array([1, 1])

	xy = np.zeros((frames_tot, 2))  # this is for the final image, which is 2D!
	dxy = np.zeros((frames_tot, 2))
	# rotation = np.zeros((frames_tot,))
	# stns_t = np.linspace(0.99, 0.2, num=frames_tot)

	'''Only for wave 2. 
	TODO: stns_t affects whole wave in the same way. Only way to get the big one is by 
	using zx mesh. The mesh is just a heatmap that should reflect the reef.'''
	# stns_t = np.log(np.linspace(start=1.0, stop=5, num=frames_tot))
	beta_pdf = beta.pdf(x=np.linspace(0, 1, frames_tot), a=10, b=50, loc=0)
	stns_t = min_max_normalization(beta_pdf, y_range=[0, 4])

	x = gi['ld'][0]
	z = gi['ld'][1]  # (formerly this was called y, but its just left_offset and y is the output done below)

	N = 0
	if P.COMPLEXITY == 0:
		N = 2
	elif P.COMPLEXITY == 1:
		N = 3

	for w in range(0, N):  # NUM WAVES

		'''
		When lam is high it means that k is low, 
		When k is low it means stn is high. 
		stn is the multiplier for y 
		'''

		if w == 0:  # OBS ADDIND WAVES LEADS TO WAVE INTERFERENCE!!!
			# d = np.array([0.2, -0.8])  # OBS this is multiplied with x and z, hence may lead to large y!
			d = np.array([0.4, -0.6])  # OBS this is multiplied with x and z, hence may lead to large y!
			c = 0.1  # [0.1, 0.05] prop to FPS EVEN MORE  from 0.2 at 20 FPS to. NEXT: Incr frames_tot for o2 AND o1
			lam = 300
			# stn0 = stn_particle
			k = 2 * np.pi / lam  # wavenumber
			# stn_particle = 0.01
			stn = stn_particle / k
			# steepness_abs = 1.0
		elif w == 1:
			d = np.array([0.2, -0.6])
			c = -0.06  # [-0.03, -0.015]
			lam = 800
			k = 2 * np.pi / lam
			stn = 0
			# steepness_abs = 1
		elif w == 2:
			d = np.array([-0.2, -0.6])
			c = 0.08  # [0.06, 0.03]
			lam = 100
			k = 2 * np.pi / lam  # wavenumber
			# stn = stn_particle / k
			stn = 0.99 / k

		for i in range(0, frames_tot):  # could probably be replaced with np or atleast list compr

			if w == 1:
				stn = (stn_particle + stns_t[i]) / k

			# stn = stns_t[i]
			y = k * np.dot(d, np.array([x, z])) - c * i  # VECTORIZE uses x origin?

			if w != 2:  # SMALL ONES MOVE LEFT
				xy[i, 0] += stn * np.cos(y)  # this one needs fixing due to foam
			elif w == 2:  # small ones
				xy[i, 0] -= stn * np.cos(y)

			xy[i, 1] += stn * np.sin(y)

			'''
			All of these are gradients, first two are just decomposed into x y
			Needed to get f direction. 
			'''
			dxy[i, 0] += 1 - stn * np.sin(y)  # mirrored! Either x or y needs to be flipped
			dxy[i, 1] += stn * np.cos(y)
			# dxy[i, 2] += (stn * np.cos(y)) / (1 - stn * np.sin(y))  # gradient: not very useful cuz it gets inf at extremes

	dxy[:, 0] = -dxy[:, 0]
	dxy[:, 1] = -dxy[:, 1]

	# dxy[:, 0] = min_max_normalization(dxy[:, 0], y_range=[-np.pi / 2, np.pi / 2])
	# # dxy[:, 0] = min_max_normalization(dxy[:, 0], y_range=[0.01, 2 * np.pi])
	# dxy[:, 1] = min_max_normalization(dxy[:, 1], y_range=[-np.pi / 2, np.pi / 2])
	# dxy[:, 2] = min_max_normalization(dxy[:, 2], y_range=[-0.99, 0.99])

	# alphas = np.full(shape=(len(dxy),), fill_value=left_start / np.pi)  # left_start ONLY affects o1
	# alphas = np.linspace(start=0.01, stop=0.99, num=frames_tot)

	'''NEED TO SHIFT BY LEFT START SOMEHOW'''
	peaks = scipy.signal.find_peaks(xy[:, 1])[0]
	alphas = np.full(shape=(len(xy),), fill_value=0.5)

	peaks_pos_y = []  # crest
	for i in range(len(peaks)):
		pk_ind = peaks[i]
		if pk_ind > 5 and xy[pk_ind, 1] > 0:
			peaks_pos_y.append(pk_ind)

	for i in range(len(peaks_pos_y) - 1):
		peak_ind0 = peaks_pos_y[i]
		peak_ind1 = peaks_pos_y[i + 1]
		num = int((peak_ind1 - peak_ind0) / 2)
		start = peak_ind0 + int(0.5 * num)
		# alphas[pk_ind0:pk_ind1 + num]

		alpha_mask = beta.pdf(x=np.linspace(0, 1, num), a=2, b=2, loc=0)
		alpha_mask = min_max_normalization(alpha_mask, y_range=[0.5, 1])
		alphas[start:start + num] = alpha_mask


	adf = 6

	# alphas = -xy[:, 0] - xy[:, 1] #+ dxy[:, 1]   # origin is left bottom
	# alphas = dxy[:, 1]
	# alphas = min_max_normalization(alphas, y_range=[0.1, 0.99])

	#

	if P.COMPLEXITY == 0:
		rotation = np.zeros(shape=(len(xy),))
	elif P.COMPLEXITY == 1:
		rotation = min_max_normalization(-xy[:, 1], y_range=[-0.1 * np.pi, 0.1 * np.pi])
		# rotation = min_max_normalization(-xy[:, 1], y_range=[-0.0001 * np.pi, 0.0001 * np.pi])

	return xy, dxy, alphas, rotation, peaks


def foam_b(o1, peak_inds):
	"""

	"""

	xy_t = np.copy(o1.xy_t)
	rotation = np.zeros((len(o1.xy),))
	alphas = np.zeros(shape=(len(xy_t),))

	for i in range(len(peak_inds) - 1):
		peak_ind0 = peak_inds[i]
		peak_ind1 = peak_inds[i + 1]

		num = int((peak_ind1 - peak_ind0) / 2)  # num is HALF

		start = int(peak_ind0 + 0.2 * num)

		# mult_x = - beta.pdf(x=np.linspace(0, 1, num), a=2, b=5, loc=0)
		# mult_x = min_max_normalization(mult_x, y_range=[0.2, 1])
		# aa = mult_x
		#
		# mult_y = beta.pdf(x=np.linspace(0, 1, num), a=2, b=5, loc=0)
		# mult_y = min_max_normalization(mult_y, y_range=[1, 1])
		#
		# xy_t[start:start + num, 0] *= mult_x
		# xy_t[start:start + num, 1] *= mult_y

		alpha_mask = beta.pdf(x=np.linspace(0, 1, num), a=2, b=3, loc=0)
		alpha_mask = min_max_normalization(alpha_mask, y_range=[0, 0.8])

		alphas[start:start + num] = alpha_mask

	aa = 7

	# PEND DEL
	# _s.alphas = np.ones(shape=(_s.gi['frames_tot']))
	# xy = np.copy(o1.xy)
	# xy[:, 0] *= 1
	# # xy[:, 1] += 100
	#
	# # alphas = np.zeros(shape=(o1.gi['frames_tot']))
	# # peaks_inds = scipy.signal.find_peaks(o1.xy[:, 1], distance=50)[0]
	#
	# # alphas = -0.2 * xy[:, 0] #+ 0.8 * xy[:, 1] #+ dxy[:, 1]   # origin is left bottom
	# # alphas = 0.2 * xy[:, 1] #+ 0.8 * xy[:, 1] #+ dxy[:, 1]   # origin is left bottom
	# a_x = o1.xy_t[:, 0]
	# inds = np.where(a_x < 0)[0]
	# a_x[inds] = 0
	#
	# a_y = o1.xy_t[:, 1]
	# # inds = np.where(a_x < 0)[0]
	# # a_x[inds] = 0
	#
	# alphas = 0.0 * a_x + 0.99 * a_y
	# # alphas = xy[:, 0]  #+ dxy[:, 1]   # origin is left bottom
	# alphas = min_max_normalization(alphas, y_range=[0.01, 0.3])
	# # alphas = min_max_normalization(alphas, y_range=[0.1, 0.99])
	# # alphas = np.ones(shape=(len(xy),))

	# xy = np.copy(xy_t)

	# shift = np.full(shape=(xy.shape), fill_value=[o1.xy[0, 0], o1.xy[0, 1]])  # THIS WILL CHANGE TO START AT INDEX (prob)
	# shift = np.full(shape=(xy_t.shape), fill_value=[o1.xy[0, 0], o1.xy[0, 1]])  # THIS WILL CHANGE TO START AT INDEX (prob)
	# xy[:, :] += shift

	# alphas[peaks_inds] = 1

	return xy_t, alphas, rotation


def foam_f(o1, peak_inds):
	"""
	TODO: Need to do projectile motion until y tangent first becomes positive.
	Precompute peaks and use those indicies as starting points.
	"""

	xy_t = np.copy(o1.xy_t)
	rotation = np.zeros((len(o1.xy),))
	alphas = np.full(shape=(len(xy_t),), fill_value=0.0)

	# pos_inds_x = np.where(xy_t[:, 0] > 0)[0]  # these are used for a later reset
	# neg_inds_x = np.where(xy_t[:, 0] < 0)[0]  # these are used for a later reset
	# pos_inds_y = np.where(xy_t[:, 1] > 0)[0]
	# neg_inds_y = np.where(xy_t[:, 1] < 0)[0]

	# aa = np.where((xy_t[:, 0] > 0) & (xy_t[:, 1] > 0))[0]
	# xy_t[aa, 0] *= 2

	'''Shift to upper left'''
	xy_t_min_x = np.min(xy_t[:, 0])
	xy_t_min_y = np.min(xy_t[:, 1])

	# xy_t[:, 0] += abs(xy_t_min_x)  # 1
	# xy_t[:, 1] += abs(xy_t_min_y)
	#
	# xy_t[:, 0] *= 3  # 2
	# xy_t[:, 1] *= 2  # 2
	#
	# xy_t[:, 0] -= abs(xy_t_min_x) * 3
	# xy_t[:, 1] -= abs(xy_t_min_y) * 2

	# span_range = np.arange(start=np.min(xy_t[:, 1]), stop=np.max(xy_t[:, 1]), dtype=int)
	# mults_x = np.linspace(start=1, stop=2, num=len(span_range))
	#
	# for i in range(len(xy_t)):
	# 	y_val_at_i = xy_t[i, 1]
	# 	qr = 5
	#
	# adf = 6

	for i in range(len(peak_inds) - 1):
		peak_ind0 = peak_inds[i]
		peak_ind1 = peak_inds[i + 1]

		num = int((peak_ind1 - peak_ind0) / 2)
		start = int(peak_ind0 + 0.0 * num)

		'''scaling'''
		mult_x = beta.pdf(x=np.linspace(0, 1, num), a=1.5, b=1.5, loc=0)
		mult_x = min_max_normalization(mult_x, y_range=[1, 2])

		mult_y = -beta.pdf(x=np.linspace(0, 1, num), a=1.5, b=1.5, loc=0)
		mult_y = min_max_normalization(mult_y, y_range=[0.5, 1])

		xy_t[start:start + num, 0] *= mult_x
		#
		# # aa = xy_t[start:start + num, 0]
		# # bb = o1.xy_t[start:start + num, 0]
		# aa = np.copy(xy_t)

		xy_t[start:start + num, 1] *= mult_y

		alpha_mask = beta.pdf(x=np.linspace(0, 1, num), a=4, b=20, loc=0)
		alpha_mask = min_max_normalization(alpha_mask, y_range=[0.0, 0.99])

		alphas[start:start + num] = alpha_mask

		aa = 5

	# xy_t[pos_inds_x, 0] *= 2
	# xy_t[neg_inds_x, 0] = o1.xy_t[neg_inds_x, 0]
	# xy_t[neg_inds_y, 1] = o1.xy_t[neg_inds_y, 1]

	# shift = np.full(shape=(o1.xy.shape), fill_value=[o1.xy[o1f_f.gi['init_frames'][0], 0],
	#                                                 o1.xy[o1f_f.gi['init_frames'][0], 1]])  # THIS WILL CHANGE TO START AT INDEX (prob)
	#
	# xy = np.copy(xy_t)
	# xy[:, :] += shift
	# xy += xy_t

	# alphas = beta.pdf(x=np.linspace(0, 1, o1f_f.gi['frames_tot']), a=2, b=10, loc=0)
	# alphas = min_max_normalization(alphas, y_range=[0.99, 0.99])

	# o1.xy[i_f:i_f + len(xy_f), 1] -= 0.5  # not gonna work here cuz this is only gonna affect next wave. SOLUTION: use steepness

	# if o1.id != '15_b_0':2
	# 	alphas = np.zeros(shape=(o1f_f.gi['frames_tot'],))

	# PEND DEL
	# thetas = np.linspace(0, 10*np.pi, num=50)
	# thetas = np.linspace(0.6 * np.pi, -1 * np.pi, num=o1f_f.gi['frames_tot'])
	# add_x = np.linspace(50, 500, num=o1f_f.gi['frames_tot'])  # TODO
	# add_y = np.linspace(-10, -200, num=o1f_f.gi['frames_tot'])  # # minus is DOWN
	# radiuss = np.linspace(25, 1, num=o1f_f.gi['frames_tot'])
	# radiuss = beta.pdf(x=np.linspace(0, 1, o1f_f.gi['frames_tot']), a=2, b=5, loc=0)
	# radiuss = min_max_normalization(radiuss, y_range=[5, 50])

	# for theta in np.linspace(0, 10*np.pi):
	# for i in range(o1f_f.gi['frames_tot']):
	# 	theta = thetas[i]
	#
	# 	# r = o1.gi['frames_tot'] - i  # radius
	# 	r = radiuss[i]  # displacement per frame
	#
	# 	# xy_f[i, 0] = r * np.cos(theta) #+ add_x[i]
	# 	# xy_f[i, 1] = r * np.sin(theta) #+ add_y[i]
	#
	# 	# y = gi['v'] * np.sin(gi['theta']) * t_lin - 0.5 * G * t_lin ** 2

	return xy_t, alphas, rotation


def shift_wave(xy_t, origin=None, gi=None):
	"""
	OBS N6 = its hardcoded for sp
	shifts it to desired xy
	y is flipped because 0 y is at top and if flip_it=True
	"""

	xy = copy.deepcopy(xy_t)

	'''x'''
	xy[:, 0] += origin[0]  # OBS THIS ORIGIN MAY BE BOTH LEFT AND RIGHT OF 640

	'''
	y: Move. y_shift_r_f_d is MORE shifting downward (i.e. positive), but only the latter portion 
	of frames is shown.
	'''
	xy[:, 1] += origin[1]

	return xy


if __name__ == '__main__':  # cant be done in trig funcs main cuz circular import
	fig = plt.figure(figsize=(10, 5))
	gi = {}
	gi['ld'] = [0, 0]
	gi['steepness'] = 150
	gi['frames_tot'] = 300
	xy, alphas = gerstner_waves(gi=gi)

	ax1 = plt.plot(alphas)
	# ax1 = plt.plot(xy[:, 0])
	ax2 = plt.plot(xy[:, 1])  # obs flipped!!!
	plt.show()



# OLD WAVE


	# Y = np.zeros((num_frames,))
	# X[0] = 500
	# xy[0, :] = [600, 400]
	# inp_x = np.arange(0, num_frames)
	# inp_x = np.arange(0, num_frames)
	# for i in range(len(xy) - 1):
	# 	# X[i] = get_x(i, 50, 10)
	# 	# X[i + 1] = get_x(X[i], 10, i)
	# 	xy[i + 1, 0] = get_x(xy[i, 0], xy[i, 1], i)
	# 	xy[i + 1, 1] = get_y(xy[i, 0], xy[i, 1], i)
	#
	# 	aa = 5
	#
	# A = np.arange(0, 100)  # X
	# B = np.arange(0, 100)  # Y

	# if gi == None:
	# 	xy = None
	# 	return None, X
	# else:
	# 	xy = np.zeros((gi['frames_tot'], 2))  # MIDPOINT
	# 	xy[:, 0] = X
		# xy[:, 1] = 400
	#
	# lam = 100
	# k = 2 * np.pi / lam
	# c = 100
	#
	# # x = np.zeros(shape=(len(xy),))
	# a = np.arange(200, 400)
	# b = np.arange(400, 600)
	# # t = np.arange(0, 200)
	# # t = np.full(shape=(200,), fill_value=1)
	# t = np.arange(0, frames_tot)
	#
	# # f = k * (p.x - _Speed * _Time.y)
	# #
	# # x = a + (np.exp(k * b) / k) * np.sin(k * (a + c * t))
	# # y = b - (np.exp(k * b) / k) * np.cos(k * (a + c * t))
	#
	# # f = k * (p.x - _Speed * _Time.y)
	# # y = 10 * np.sin(k * (a))
	# y = 10 * np.sin(k * (a - c * t))
	# # x = np.sin(k * (a + c * t))
	# # y = -np.cos(k * (a + c * t))
	#
	# xy[:, 0] = a
	# xy[:, 1] = y

	# if gi != None: