import numpy as np
import copy
import matplotlib.pyplot as plt

import P
from src.trig_functions import min_max_normalization
import random
import scipy
from scipy.stats import beta, gamma


def gerstner_waves(o1, o0):
	"""
	Per particle!
	3 waves:
	0: The common ones
	1: The big one
	2: The small ones in opposite direction

	OBS: xy IN HERE IS ACTUALLY xy_t
	"""

	# lam = 1.5  # np.pi / 2 - 0.07  # pi is divided by this, WAVELENGTH
	# lam = 200  # np.pi / 2 - 0.07  # pi is divided by this, WAVELENGTH, VERY SENSITIVE

	# c = 0.5
	# c = -np.sqrt(9.8 / k)
	# stn_particle = gi['steepness']  # need beta dists over zx mesh

	# left_start = gi['o1_left_start']

	frames_tot = o1.gi['frames_tot']

	d = np.array([None, None])

	xy = np.zeros((frames_tot, 2))  # this is for the final image, which is 2D!
	dxy = np.zeros((frames_tot, 2))
	rotation = np.zeros((frames_tot,))
	scale = np.ones((frames_tot,))

	xy0 = np.zeros((frames_tot, 2))
	xy1 = np.zeros((frames_tot, 2))
	xy2 = np.zeros((frames_tot, 2))
	dxy0 = np.zeros((frames_tot, 2))
	dxy1 = np.zeros((frames_tot, 2))
	dxy2 = np.zeros((frames_tot, 2))
	peaks0 = np.zeros((frames_tot,))
	peaks1 = np.zeros((frames_tot,))
	peaks2 = np.zeros((frames_tot,))

	# y_only_2 = np.zeros((frames_tot,))

	# stns_t = np.linspace(0.99, 0.2, num=frames_tot)

	'''Only for wave 2. 
	TODO: stns_t affects whole wave in the same way. Only way to get the big one is by 
	using zx mesh. The mesh is just a heatmap that should reflect the reef.'''
	# stns_t = np.log(np.linspace(start=1.0, stop=5, num=frames_tot))
	beta_pdf = beta.pdf(x=np.linspace(0, 1, frames_tot), a=10, b=50, loc=0)
	stns_t = min_max_normalization(beta_pdf, y_range=[0, 1.7])  # OBS when added = interference

	x = o1.gi['ld'][0]
	z = o1.gi['ld'][1]  # (formerly this was called y, but its just left_offset and y is the output done below)



	SS = [0, 1, 2]
	SS = [0]
	# SS = [1]
	# SS = [2]
	# SS = [0, 2]
	# if P.COMPLEXITY == 0:
	# 	SS = [0, 3]
	# elif P.COMPLEXITY == 1:
	# 	SS = [0, 3]

	for w in SS:  # NUM WAVES

		'''
		When lam is high it means that k is low, 
		When k is low it means stn is high. 
		stn is the multiplier for y
		
		OBS ADDIND WAVES LEADS TO WAVE INTERFERENCE!!! 
		Perhaps not? Increasing d will definitely increase k  
		'''

		if w == 0:  #
			d = np.array([0.2, -0.8])  # OBS this is multiplied with x and z, hence may lead to large y!
			# d = np.array([0.4, -0.9])  # OBS this is multiplied with x and z, hence may lead to large y!
			# d = np.array([0.9, -0.1])  # OBS this is multiplied with x and z, hence may lead to large y!
			c = 0.1  # [0.1, 0.02] prop to FPS EVEN MORE  from 0.2 at 20 FPS to. NEXT: Incr frames_tot for o2 AND o1
			if P.COMPLEXITY == 1:
				c /= 5
				d = np.array([0.2, -0.8])  # OBS this is multiplied with x and z, hence may lead to large y!
				# d = np.array([0.9, -0.1])  # OBS this is multiplied with x and z, hence may lead to large y!
			lam = 200
			# stn0 = stn_particle
			k = 2 * np.pi / lam  # wavenumber
			# stn_particle = 0.01
			stn_particle = o0.gi.stns_zx0[o1.z_key, o1.x_key]
			stn = stn_particle / k
		# steepness_abs = 1.0
		elif w == 1:  # BIG ONE
			d = np.array([0.4, -0.6])
			# d = np.array([0.9, -0.1])
			# c = 0.1  # [-0.03, -0.015] ?????
			c = 0.1  # [0.1, 0.02]
			if P.COMPLEXITY == 1:
				c /= 5
			lam = 1200  # Basically, there are many waves, but only a few will be amplified a lot due to stns_t
			k = 2 * np.pi / lam
			stn_particle = o0.gi.stns_zx1[o1.z_key, o1.x_key]
			stn = None  # cuz its also affected by time
		# steepness_abs = 1
		elif w == 2:
			d = np.array([-0.2, -0.7])
			# c = 0.1  # [0.06, 0.03]
			c = 0.1  # [0.1, 0.02]
			if P.COMPLEXITY == 1:
				c /= 5
			lam = 80
			k = 2 * np.pi / lam  # wavenumber
			# stn = stn_particle / k
			stn = 1 / k

		for i in range(0, frames_tot):  # could probably be replaced with np or atleast list compr

			if w == 1:
				stn = (0.4 * stn_particle + 0.6 * stns_t[i]) / k
			# stn = stn_particle / k

			y = k * np.dot(d, np.array([x, z])) - c * i  # VECTORIZE uses x origin? Also might have to use FFT here

			if w != 2:  # SMALL ONES MOVE LEFT
				xy[i, 0] += stn * np.cos(y)  # this one needs fixing due to foam
			elif w == 2:  # small ones
				xy[i, 0] -= stn * np.cos(y)

			xy[i, 1] += stn * np.sin(y)

			if w == 0:
				xy0[i, 0] = stn * np.cos(y)
				xy0[i, 1] = stn * np.sin(y)
				dxy0[i, 0] = 1 - stn * np.sin(y)
				dxy0[i, 1] = stn * np.cos(y)
			if w == 1:
				xy1[i, 0] = stn * np.cos(y)
				xy1[i, 1] = stn * np.sin(y)
				dxy1[i, 0] = 1 - stn * np.sin(y)
				dxy1[i, 1] = stn * np.cos(y)
			if w == 2:
				xy2[i, 0] = stn * np.cos(y)
				xy2[i, 1] = - stn * np.sin(y)
				dxy2[i, 0] = 1 - stn * np.sin(y)  # CHECK IT!!!
				dxy2[i, 1] = stn * np.cos(y)  # CHECK IT!!!

			# if w == 2:  # to ensure foam for 2. Perhaps?
			# 	y_only_2[i] = stn * np.sin(y)

			'''
			All of these are gradients, first two are just decomposed into x y
			Needed to get f direction. 
			'''
			dxy[i, 0] += 1 - stn * np.sin(y)  # mirrored! Either x or y needs to be flipped
			dxy[i, 1] += stn * np.cos(y)
			# dxy[i, 2] += (stn * np.cos(y)) / (1 - stn * np.sin(y))  # gradient: not very useful cuz it gets inf at extremes

			if w in [0, 1]:  # MIGHT NEED SHIFTING
				# rotation[i] += dxy[i, 1]
				rotation[i] = dxy[i, 1]

			scale[i] = - np.sin(y)

	dxy[:, 0] = -dxy[:, 0]
	dxy[:, 1] = -dxy[:, 1]

	'''Used below by alpha'''
	peaks = scipy.signal.find_peaks(xy[:, 1])[0]  # includes troughs
	peaks_pos_y = []  # crest
	for i in range(len(peaks)):  # could be replaced with lambda prob
		pk_ind = peaks[i]
		if pk_ind > 5 and xy[pk_ind, 1] > 0:  # check that peak y value is positive
			peaks_pos_y.append(pk_ind)

	'''ALPHA THROUGH TIME OBS ONLY FOR STATIC'''
	ALPHA_LOW_BOUND = 0.5
	ALPHA_UP_BOUND = 1
	alphas = np.full(shape=(len(xy),), fill_value=ALPHA_LOW_BOUND)

	for i in range(len(peaks_pos_y) - 1):
		peak_ind0 = peaks_pos_y[i]
		peak_ind1 = peaks_pos_y[i + 1]
		# num = int((peak_ind1 - peak_ind0) / 2)
		# start = peak_ind0 + int(0.5 * num)
		num = int((peak_ind1 - peak_ind0))
		start = peak_ind0
		# alphas[pk_ind0:pk_ind1 + num]

		# alphas_tp = np.sin(np.linspace(0, -0.5 * np.pi, num=int(peak_ind1 - peak_ind0)))

		# alpha_mask_t = -beta.pdf(x=np.linspace(0, 1, num), a=2, b=2, loc=0)
		alpha_mask_t = np.sin(np.linspace(0, np.pi, num=int(peak_ind1 - peak_ind0)))
		alpha_mask_t = min_max_normalization(alpha_mask_t, y_range=[ALPHA_LOW_BOUND, ALPHA_UP_BOUND])  # [0.5, 1]
		alphas[peak_ind0:peak_ind1] = alpha_mask_t

	if P.COMPLEXITY == 0:
		rotation = np.zeros(shape=(len(xy),))  # JUST FOR ROUND ONES
	elif P.COMPLEXITY == 1:
		'''T&R More neg values mean more counterclockwise'''
		if len(SS) > 1:
			if SS[0] == 2 and SS[1] == 3:  # ????
				pass
			else:
				rotation = min_max_normalization(rotation, y_range=[-0.2 * np.pi, 0.2 * np.pi])
		else:
			rotation = min_max_normalization(rotation, y_range=[-0.6, 0.6])

	# scale = min_max_normalization(scale, y_range=[1, 1.3])
	scale = min_max_normalization(scale, y_range=[0.99, 1.2])

	return xy, dxy, alphas, rotation, peaks, xy0, dxy0, xy1, dxy1, xy2, dxy2, scale


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

		start = int(peak_ind0 + 0.0 * num)

		# mult_x = - beta.pdf(x=np.linspace(0, 1, num), a=2, b=5, loc=0)
		# mult_x = min_max_normalization(mult_x, y_range=[0.2, 1])
		# aa = mult_x
		#
		# mult_y = beta.pdf(x=np.linspace(0, 1, num), a=2, b=5, loc=0)
		# mult_y = min_max_normalization(mult_y, y_range=[1, 1])
		#
		# xy_t[start:start + num, 0] *= mult_x
		# xy_t[start:start + num, 1] *= mult_y

		alpha_mask = beta.pdf(x=np.linspace(0, 1, num), a=4, b=20, loc=0)
		alpha_mask = min_max_normalization(alpha_mask, y_range=[0, 0.8])

		alphas[start:start + num] = alpha_mask

	return xy_t, alphas, rotation


def foam_f(o1):
	"""
	peak_inds: they are now just imported from static
	New idea: Everything between start and start + num is available.
	So use everything and then just move object to next peak by shift.
	Instead of setting alpha as beta with early peak, set peak to middle and

	"""

	xy_t = np.copy(o1.xy_t)
	xy_t0 = np.copy(o1.xy_t0)
	dxy0 = np.copy(o1.dxy0)

	# rotation0 = np.copy(o1.dxy0[:, 1])

	rotation0 = np.zeros((len(o1.xy),))  # CALCULATED HERE
	alphas = np.full(shape=(len(xy_t),), fill_value=0.0)

	'''OBS height SUPER IMPORTANT TO AVOID 2 GETTING f '''
	# peak_inds = scipy.signal.find_peaks(xy_t[:, 1], height=20, distance=50)[0]  # OBS 20 needs tuning!!!
	# peak_inds = scipy.signal.find_peaks(xy_t[:, 1], height=15, distance=10)[0]  # OBS 20 needs tuning!!!
	peak_inds = scipy.signal.find_peaks(xy_t0[:, 1], height=25, distance=10)[0]  # OBS 20 needs tuning!!!
	'''Below DEPRECATED probably need to be PERCENTAGE'''
	peak_inds -= 10  # neg mean that they will start before the actual peak
	neg_inds = np.where(peak_inds < 0)[0]
	if len(neg_inds) > 0:  # THIS IS NEEDED DUE TO peak_inds -= 10
		peak_inds[neg_inds] = 0

	'''MOVE THIS TO INSIDE LOOP'''
	# if len(peak_inds) > 1:  # First ind will turn neg for some points
	# 	if peak_inds[0] < 10:
	# 		peak_inds = peak_inds[1:]
	# 	peak_inds -= 10
	# 	if len(np.where(peak_inds < 0)[0]) > 0:
	# 		raise Exception("THIS IS NEEDED DUE TO peak_inds -= 10")

	# PEND DEL: its done in loop below instead
	# pos_inds_x = np.where(xy_t[:, 0] > 0)[0]  # these are used for a later reset
	# neg_inds_x = np.where(xy_t[:, 0] < 0)[0]  # these are used for a later reset
	# pos_inds_y = np.where(xy_t[:, 1] > 0)[0]
	# neg_inds_y = np.where(xy_t[:, 1] < 0)[0]

	# aa = np.where((xy_t[:, 0] > 0) & (xy_t[:, 1] > 0))[0]
	# xy_t[aa, 0] *= 2

	# xy_t_min_x = np.min(xy_t[:, 0])
	# xy_t_min_y = np.min(xy_t[:, 1])

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
		# mid_ind = int(peak_ind1 - peak_ind0)

		'''OBS THIS WRITES TO xy_t STRAIGHT'''
		xy_tp = xy_t[peak_ind0:peak_ind1]  # xy_tp: xy coords time between peaks
		xy_tp0 = xy_t0[peak_ind0:peak_ind1]  # xy_tp: xy coords time between peaks

		rotation_tp0 = np.sin(np.linspace(0, -0.5 * np.pi, num=int(peak_ind1 - peak_ind0)))

		# rotation_tp0 = min_max_normalization(rotation_tp0, y_range=[-0.2 * np.pi, 0.2 * np.pi])
		# rotation_tp0 = min_max_normalization(rotation_tp0, y_range=[-0.2 * np.pi, 0.2 * np.pi])
		# rotation_tp0 = np.sin(rotation_tp0)
		rotation0[peak_ind0:peak_ind1] = rotation_tp0


		'''
		Generating the break motion by scaling up the Gersner rotation
		Might also need to shift it. Which is fine if alpha used correctly
		
		New thing: Instead of multiplying Gerstner circle with constant, 
		its much cleaner to extract v at top of wave and then generating a projectile motion. 
		BUT, this only works for downward motion
		
		'''

		x_max_ind = np.argmax(xy_tp[:, 0])
		y_min_ind = np.argmin(xy_tp[:, 1])
		y_max_ind = np.argmax(xy_tp0[:, 1])

		start_x = xy_tp[y_max_ind, 0]
		start_y = xy_tp[y_max_ind, 1]

		# if y_max_ind + 1 < len(xy_tp):
		# 	v_frame = xy_tp[y_max_ind + 1, 0] - xy_tp[y_max_ind, 0]
		# elif y_max_ind > 0:
		# 	v_frame = xy_tp[y_max_ind, 0] - xy_tp[y_max_ind - 1, 0]
		# else:
		# 	v_frame = 1.7
		v_frame = (xy_tp0[1, 0] - xy_tp0[0, 0]) * 0.4
		# v_frame = xy_tp[1, 0] - xy_tp[0, 0]
		# v_frame = o1.o0.gi.stns_zx0[o1.z_key, o1.x_key]
		# v_frame = 1.7
		# v_frame = 2
		if P.COMPLEXITY == 1:
			v_frame = 0.35

		# v = np.max(xy_tp[:, 1]) + abs(np.min(xy_tp))
		# v = 1.7

		'''
		NUM HERE IS FOR PROJ. STARTS WHEN Y AT MAX
		TODO: NUM SHOULD BE SPLIT INTO TWO PARTS
		FIRST NUM IS ONLY PROJ
		SECOND NUM IS FOR RISING
		'''
		num = len(xy_tp) - y_max_ind  # OBS! This is where proj motion is used
		xy_proj = np.zeros(shape=(num, 2))
		v = v_frame * num
		theta = 0
		G = 9.8
		h = (np.max(xy_tp[:, 1]) + abs(np.min(xy_tp))) * 2  # more, = more fall ALSO TO RIGHT
		# t_flight = (v * np.sin(theta) + np.sqrt((v * np.sin(theta)) ** 2 + 2 * G * h)) / G
		t_flight = (np.sqrt(2 * G * h)) / G

		t_lin = np.linspace(0, t_flight, num)
		xy_proj[:, 0] = v * np.cos(theta) * t_lin
		xy_proj[:, 1] = v * np.sin(theta) * 2 * t_lin - 0.5 * G * t_lin ** 2

		xy_proj[:, 0] += start_x
		xy_proj[:, 1] += start_y

		xy_tp0[y_max_ind:, :] = xy_proj


		# x_end = xy_tp[len(xy_tp) * 1.7

		# mult_x = np.linspace(start=1, stop=2, num=len(xy_tp))
		# mult_y = -beta.pdf(x=np.linspace(0, 1, num=len(xy_tp)), a=1.5, b=1.5, loc=0)
		# mult_y = min_max_normalization(mult_y, y_range=[0.5, 1])
		#
		# xy_tp[:, 0] *= mult_x
		# xy_tp[:, 1] *= mult_y
		#
		# x_max_ind = np.argmax(xy_tp[:, 0])
		# y_min_ind = np.argmin(xy_tp[:, 1])
		#
		# xy_tp[x_max_ind:, 0] = xy_tp[x_max_ind, 0]
		#
		# y_desc = np.linspace(start=xy_tp[x_max_ind, 1], stop=xy_tp[y_min_ind, 1], num=len(xy_tp) - x_max_ind)
		# xy_tp[x_max_ind:, 1] = y_desc
		#
		# xy_t[peak_ind0:peak_ind1, :] = xy_tp

		# alpha_mask_t = beta.pdf(x=np.linspace(0, 1, num), a=4, b=20, loc=0)

		# peak_ind0a = peak_ind0
		# len_extra = 0
		# if peak_ind0a > 20:
		# 	peak_ind0a -= 20
		# 	len_extra = 20

		alpha_UB = 1
		# PEND DELif h < 300 and P.COMPLEXITY:
		# 	alpha_UB = 0.1

		# alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp)), a=2, b=1.8, loc=0)  # OBS THESE INCLUDE EVERYTHING
		alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp)), a=1.5, b=2.5, loc=0)  # ONLY FIRST PART
		alpha_mask_t = min_max_normalization(alpha_mask_t, y_range=[0.0, alpha_UB])

		# if peak_ind0 > 20:
		# 	alphas[peak_ind0 - 20:peak_ind1 - 20] = alpha_mask_t
		# else:
		alphas[peak_ind0:peak_ind1] = alpha_mask_t

	# rotation0 = min_max_normalization(rotation0, y_range=[-0.2 * np.pi, 0.2 * np.pi])

	# bb = 5

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

	# PEND DEL NO!: Its too complicated to create a new circle from nothing perhaps.
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

	return xy_t0, alphas, rotation0


# def shift_wave(xy_t, origin=None, gi=None):
# 	"""
# 	OBS N6 = its hardcoded for sp
# 	shifts it to desired xy
# 	y is flipped because 0 y is at top and if flip_it=True
# 	"""
#
# 	xy = copy.deepcopy(xy_t)
#
# 	'''x'''
# 	xy[:, 0] += origin[0]  # OBS THIS ORIGIN MAY BE BOTH LEFT AND RIGHT OF 640
#
# 	'''
# 	y: Move. y_shift_r_f_d is MORE shifting downward (i.e. positive), but only the latter portion
# 	of frames is shown.
# 	'''
# 	xy[:, 1] += origin[1]
#
# 	return xy
