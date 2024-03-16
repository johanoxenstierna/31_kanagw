
import numpy as np
import copy
import matplotlib.pyplot as plt
from src.trig_functions import min_max_normalization
import random
import scipy
from scipy.stats import beta, gamma


def gerstner_wave(gi):

	"""
	Per particle! o2
	"""

	lam = 1.5  # np.pi / 2 - 0.07  # pi is divided by this, WAVELENGTH
	lam = 200  # np.pi / 2 - 0.07  # pi is divided by this, WAVELENGTH, VERY SENSITIVE

	c = 0.5
	# c = -np.sqrt(9.8 / k)
	steepness_abs = gi['steepness']

	# left_start = gi['o1_left_start']

	frames_tot = gi['frames_tot']

	d = np.array([1, 1])

	xy = np.zeros((frames_tot, 2))  # this is for the final image, which is 2D!
	dxy = np.zeros((frames_tot, 2))

	'''Shifting is irrelevant here, because its done in o2 finish_info'''
	x = gi['ld'][0]
	z = gi['ld'][1]  # (formerly this was called y, but its just left_offset and y is the output done below)

	# alphas = np.zeros(shape=(len(xy),))
	alphas = np.full(shape=(len(xy),), fill_value=0.5)

	for w in range(0, 1):  # NUM WAVES

		if w == 0:  # OBS ADDIND WAVES LEADS TO WAVE INTERFERENCE!!!
			d = np.array([0.6, -0.8])  # OBS this is multiplied with x and z, hence may lead to large y!
			# d = np.array([1, 0.0])  # OBS this is multiplied with x and z, hence may lead to large y!
			c = 0.1  # prop to FPS EVEN MORE  from 0.2 at 20 FPS to. NEXT: Incr frames_tot for o2 AND o1
			lam = 200
			# steepness_abs = 1.0
		elif w == 1:
			d = np.array([-0.1, -0.4])
			c = 0.1  # from 0.6 -> 0.06
			lam = 50
			# steepness_abs = 0.9

		k = 2 * np.pi / lam  # wavenumber
		stn = steepness_abs / k

		for i in range(0, frames_tot):  # could probably be replaced with np or atleast list compr

			y = k * np.dot(d, np.array([x, z])) - c * i  # VECTORIZE uses x origin?

			xy[i, 0] += stn * np.cos(y)  # this one needs fixing due to foam
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
	# alphas = -xy[:, 0] - xy[:, 1] #+ dxy[:, 1]   # origin is left bottom
	# alphas = min_max_normalization(alphas, y_range=[0.01, 0.9])

	return xy, dxy, alphas


def foam_b(o1):
	"""

	"""

	# _s.alphas = np.ones(shape=(_s.gi['frames_tot']))

	xy = np.copy(o1.xy)
	xy[:, 0] *= 1

	# alphas = np.zeros(shape=(o1.gi['frames_tot']))
	# peaks_inds = scipy.signal.find_peaks(o1.xy[:, 1], distance=50)[0]

	# alphas = -0.2 * xy[:, 0] #+ 0.8 * xy[:, 1] #+ dxy[:, 1]   # origin is left bottom
	# alphas = 0.2 * xy[:, 1] #+ 0.8 * xy[:, 1] #+ dxy[:, 1]   # origin is left bottom
	a_x = o1.xy_t[:, 0]
	inds = np.where(a_x < 0)[0]
	a_x[inds] = 0

	a_y = o1.xy_t[:, 1]
	# inds = np.where(a_x < 0)[0]
	# a_x[inds] = 0

	alphas = 0.6 * a_x + 0.4 * a_y
	# alphas = xy[:, 0]  #+ dxy[:, 1]   # origin is left bottom
	alphas = min_max_normalization(alphas, y_range=[0.01, 0.5])
	# alphas = np.ones(shape=(len(xy),))

	aa = 7

	# alphas[peaks_inds] = 1

	return xy, alphas


def foam_f(o1f_f, o1, i_f):
	"""
	TODO: Need to do projectile motion until y tangent first becomes positive.
	Precompute peaks and use those indicies as starting points.
	"""

	# xy = np.copy(o1.xy)

	xy_f = np.zeros(shape=(o1f_f.gi['frames_tot'], 2))
	# thetas = np.linspace(0, 10*np.pi, num=50)
	thetas = np.linspace(0.6 * np.pi, -2 * np.pi, num=o1f_f.gi['frames_tot'])
	add_x = np.linspace(0, 600, num=o1f_f.gi['frames_tot'])  # TODO
	sub_y = np.linspace(20, 50, num=o1f_f.gi['frames_tot'])  # TODO
	# radiuss = np.linspace(25, 1, num=o1f_f.gi['frames_tot'])
	radiuss = beta.pdf(x=np.linspace(0, 1, o1f_f.gi['frames_tot']), a=2, b=5, loc=0)
	radiuss = min_max_normalization(radiuss, y_range=[5, 30])

	# for theta in np.linspace(0, 10*np.pi):
	for i in range(o1f_f.gi['frames_tot']):
		theta = thetas[i]

		# r = o1.gi['frames_tot'] - i  # radius
		r = radiuss[i]  # displacement per frame

		xy_f[i, 0] = r * np.cos(theta) + add_x[i]
		xy_f[i, 1] = r * np.sin(theta) - sub_y[i]
		# y = gi['v'] * np.sin(gi['theta']) * t_lin - 0.5 * G * t_lin ** 2

	shift = np.full(shape=(xy_f.shape), fill_value=[o1.xy[o1f_f.gi['init_frames'][0], 0],
	                                                o1.xy[o1f_f.gi['init_frames'][0], 1]])  # THIS WILL CHANGE TO START AT INDEX (prob)

	xy_f[:, :] += shift

	alphas = radiuss
	# alphas = np.ones(shape=(len(radiuss),))
	alphas = min_max_normalization(alphas, y_range=[0.01, 0.9])

	# o1.xy[i_f:i_f + len(xy_f), 1] -= 0.5  # not gonna work here cuz this is only gonna affect next wave. SOLUTION: use steepness

	# if o1.id != '15_b_0':2
	# 	alphas = np.zeros(shape=(o1f_f.gi['frames_tot'],))

	aa = 5

	# alphas[peaks_inds] = 1

	# PROJ
	# xy = np.zeros((gi['frames_tot'], 2))  # MIDPOINT
	#
	# G = 9.8
	# # h = gi['v'] * 10  # THIS IS THE NUMBER OF PIXELS IT WILL GO
	# h = 0.7 * 400 + 0.3 * gi['v'] * 5  # THIS IS THE NUMBER OF PIXELS IT WILL GO
	#
	# # t_flight = 6 * v * np.sin(theta) / G
	#
	# '''
    # OBS since projectile is launched from a height, the calculation is different:
    # https://www.omnicalculator.com/physics/time-of-flight-projectile-motion
    # from ground level:
    # t_flight = 4 * gi['v'] * np.sin(gi['theta']) / G  # 4 means they land at origin. 5 little bit below
	#
    # '''
	# t_flight = (gi['v'] * np.sin(gi['theta']) + np.sqrt((gi['v'] * np.sin(gi['theta'])) ** 2 + 2 * G * h)) / G
	#
	# t_lin = np.linspace(0, t_flight, gi['frames_tot'])
	# # t_geo = np.geomspace(0.08, t_flight ** 1.2, gi['frames_tot'])
	# # t_geo_0 = np.geomspace(0.5, t_flight ** 1, gi['frames_tot'])  # POWER CONTROLS DISTANCE
	# # t_geo_1 = np.geomspace(0.5, t_flight ** 1, gi['frames_tot'])
	#
	# x = gi['v'] * np.cos(gi['theta']) * t_lin
	# # x_lin = abs(gi['v'] * np.cos(gi['theta']) * t_lin)  # THIS IS ALWAYS POSITIVE
	# # x_geo = abs(2 * gi['v'] * np.cos(gi['theta']) * t_geo_0)  # THIS IS ALWAYS POSITIVE. KEEP IT SIMPLE
	# # # x = 0.0001 * x_lin * t_lin + 0.2 * x_lin * x_geo
	# # # x = 0.00001 * x_lin * t_lin + 0.1 * x_lin * x_geo
	# # # x = 0.001 * x_lin * t_lin + 0.005 * x_lin * x_geo
	# # # x = 0.05 * x_lin + 0.95 * x_geo
	# # x = x
	#
	# '''If theta is close enough '''
	# y = gi['v'] * np.sin(gi['theta']) * 2 * t_lin - 0.5 * G * t_lin ** 2
	#
	# # y_lin = gi['v'] * np.sin(gi['theta']) * 2 * t_lin #- 0.5 * G * t_lin ** 2  # OBS OBS this affect both up and down equally
	# # y_geo = gi['v'] * np.sin(gi['theta']) * 2 * t_geo_1 - 0.5 * G * t_geo_1 ** 2
	#
	# # y = 0.3 * y_lin + 0.7 * y_geo  # THIS AFFECTS HOW FAR DOWN THEY GO
	# # y = 0.05 * y_lin + 0.95 * y_geo  # THIS AFFECTS HOW FAR DOWN THEY GO
	# # y = y_geo  # THIS AFFECTS HOW FAR DOWN THEY GO
	#
	# xy[:, 0] = x
	# xy[:, 1] = y
	#
	#




	return xy_f, alphas



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
	xy, alphas = gerstner_wave(gi=gi)

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