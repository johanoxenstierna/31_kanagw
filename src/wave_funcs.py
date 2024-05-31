import numpy as np
import copy
import matplotlib.pyplot as plt

import P
from src.trig_functions import min_max_normalization, min_max_normalize_array
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

	YT = np.zeros((frames_tot, P.NUM_Z, P.NUM_X), dtype=np.float16)

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
	# SS = [0, 1]

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
			# d = np.array([0.3, -0.7])  # OBS this is multiplied with x and z, hence may lead to large y!
			# d = np.array([0.8,  -0.2])  # OBS this is multiplied with x and z, hence may lead to large y!
			c = 0.15  # [0.1, 0.02] prop to FPS EVEN MORE  from 0.2 at 20 FPS to. NEXT: Incr frames_tot for o2 AND o1
			if P.COMPLEXITY == 1:
				c /= 5
				# d = np.array([0.2, -0.8])  # OBS this is multiplied with x and z, hence may lead to large y!
				# d = np.array([0.9, -0.1])  # OBS this is multiplied with x and z, hence may lead to large y!
			lam = 240  # DOES NOT AFFECT NUM FRAMES BETWEEN WAVES
			# stn0 = stn_particle
			k = 2 * np.pi / lam  # wavenumber
			# stn_particle = 0.01

		# steepness_abs = 1.0
		elif w == 1:  # BIG ONE
			d = np.array([0.25, -0.75])
			# d = np.array([0.4, -0.6])
			# d = np.array([0.9, -0.1])
			# c = 0.1  # [-0.03, -0.015] ?????
			c = 0.15  # [0.1, 0.02]
			if P.COMPLEXITY == 1:
				c /= 5
			lam = 600  # Basically, there are many waves, but only a few will be amplified a lot due to stns_t
			k = 2 * np.pi / lam
			# stn_particle = o0.gi.stns_zx1[o1.z_key, o1.x_key]
			# stn_particle = o0.gi.stns_ZX[0, o1.z_key, o1.x_key]
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

			if w == 0:
				stn_particle = o0.gi.stns_TZX[i, o1.z_key, o1.x_key]
				stn = stn_particle / k
			if w == 1:
				stn_particle = o0.gi.stns_TZX[i, o1.z_key, o1.x_key]
				# stn = (0.4 * stn_particle + 0.6 * stns_t[i]) / k
				stn = stn_particle / k
			if w == 2:
				stn_particle = o0.gi.stns_TZX[i, o1.z_key, o1.x_key]
				stn = stn_particle / k

			# stn = stn_particle / k

			y = k * np.dot(d, np.array([x, z])) - c * i  # VECTORIZE uses x origin? Also might have to use FFT here

			if w != 2:  # SMALL ONES MOVE LEFT
				xy[i, 0] += (stn * np.cos(y)) / 2  # this one needs fixing due to foam
			elif w == 2:  # small ones
				xy[i, 0] -= (stn * np.cos(y)) / 2

			xy[i, 1] += (stn * np.sin(y)) / 2
			YT[i, o1.z_key, o1.x_key] += (stn * np.sin(y)) / 2

			'''Wave specific'''
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
	ALPHA_UP_BOUND = 0.6
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

	# if P.COMPLEXITY == 0:
	# 	rotation = np.zeros(shape=(len(xy),))  # JUST FOR ROUND ONES
	# elif P.COMPLEXITY == 1:
	# 	'''T&R More neg values mean more counterclockwise'''
	# 	# if len(SS) > 1:
	# 	# 	if SS[0] == 2 and SS[1] == 3:  # ????
	# 	# 		pass
	# 	# 	else:
	# 	# 		rotation = min_max_normalization(rotation, y_range=[-0.2 * np.pi, 0.2 * np.pi])
	# 	# else:
	rotation = min_max_normalization(rotation, y_range=[-1, 1])

	# scale = min_max_normalization(scale, y_range=[1, 1.3])
	scale = min_max_normalization(scale, y_range=[0.99, 1.1])

	return xy, dxy, alphas, rotation, peaks, xy0, dxy0, xy1, dxy1, xy2, dxy2, scale, YT


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
	New idea: Everything between start and start + num is available.
	So use everything and then just move object to next peak by shift.
	S H I F T   of static. Makes sense: If wave is crazy, foam is also crazy
	"""

	EARLINESS_SHIFT = 5
	MIN_DIST_FRAMES_BET_WAVES = 15

	xy_t = np.copy(o1.xy_t)
	xy_t0 = np.copy(o1.xy_t0)

	rotation0 = np.full((len(o1.xy)), fill_value=-0.0001)  # CALCULATED HERE
	alphas = np.full(shape=(len(xy_t),), fill_value=0.0)
	scale = np.zeros(shape=(len(xy_t),))

	'''
	Peaks found using xy_t
	But v and h found using xy_t0
	'''

	# peak_inds = scipy.signal.find_peaks(xy_t[:, 1], distance=MIN_DIST_FRAMES_BET_WAVES)[0]  # OBS 20 needs tuning!!!
	peak_inds = scipy.signal.find_peaks(xy_t[:, 1], distance=MIN_DIST_FRAMES_BET_WAVES, height=20)[0]  # OBS 20 needs tuning!!!
	peak_inds -= EARLINESS_SHIFT  # neg mean that they will start before the actual peak
	neg_inds = np.where(peak_inds < 0)[0]
	if len(neg_inds) > 0:  # THIS IS NEEDED DUE TO peak_inds -= 10
		peak_inds = peak_inds[neg_inds[-1] + 1:]  # dont use neg inds

	'''
	Need to increase v with x and z. 
	Wave breaks
	Use o1 id
	
	NEW: Now that we have peaks, we can go back to using t instead of t0
	Conjecture: Have to pick EITHER tp OR tp0 below. 
	'''

	# v_mult = o1.o0.gi.vmult_zx[o1.z_key, o1.x_key]
	# h_mult = o1.o0.gi.stns_ZX[0, o1.z_key, o1.x_key]
	# stn = o1.o0.gi.stns_ZX[0, o1.z_key, o1.x_key]

	# h = o1.o0.gi.TH[0, o1.z_key, o1.x_key]
	# x_displ = 500

	for i in range(len(peak_inds) - 1):

		peak_ind0 = peak_inds[i]
		peak_ind1 = peak_inds[i + 1]

		'''OBS THIS WRITES TO xy_t STRAIGHT'''
		xy_tp = np.copy(xy_t[peak_ind0:peak_ind1])  # xy_tp: xy coords time between peaks
		xy_tp0 = np.copy(xy_t0[peak_ind0:peak_ind1])  # xy_tp: xy coords time between peaks
		h = o1.o0.gi.TH[peak_ind0, o1.z_key, o1.x_key]

		if len(xy_tp0) < MIN_DIST_FRAMES_BET_WAVES:
			raise Exception("W   T   F")

		rotation_tp = np.linspace(0, -1.5 * np.pi, num=int(peak_ind1 - peak_ind0))
		rotation_tp += np.random.uniform(-0.2, 0.4, size=1)

		rotation0[peak_ind0:peak_ind1] = rotation_tp

		scale_tp = np.linspace(0.2, 1.2, num=int(peak_ind1 - peak_ind0))
		scale[peak_ind0:peak_ind1] = scale_tp

		'''
		Generating the break motion by scaling up the Gersner rotation
		Might also need to shift it. Which is fine if alpha used correctly
		
		New thing: Instead of multiplying Gerstner circle with constant, 
		its much cleaner to extract v at top of wave and then generating a projectile motion. 
		BUT, this only works for downward motion
		'''

		# y_max_ind = int(len(xy_tp0) * 0.1)
		y_max_ind = EARLINESS_SHIFT
		y_peak0 = xy_tp[y_max_ind, 1]
		y_min_ind = np.argmin(xy_tp[:, 1])
		y_min = xy_tp[y_min_ind, 1]
		y_peak1 = xy_tp[-1, 1]
		y_fall_dist = y_peak1 - y_min

		x_max_ind = np.argmax(xy_tp[:, 0])  # DOESNT WORK WITH MULTIPLE WAVES. TODO: USE PI INSTEAD
		x_max = xy_tp[x_max_ind, 0]
		x_min_ind = np.argmin(xy_tp[:, 0])
		x_min = xy_tp[x_min_ind, 0]
		x_peak_ind1 = xy_tp[-1, 0]
		x_right_dist = x_max - x_min

		yy = o1.YT[0, 3, 3]
		asdf = 5

		'''
		NUM HERE IS FOR PROJ. STARTS WHEN Y AT MAX
		NUM SHOULD BE SPLIT INTO TWO PARTS (Maybe not)
		NUM_P IS ONLY PROJ
		NUM_B IS FOR RISING		'''

		num_p = len(xy_tp0)

		# v_frame = abs(xy_t0[y_max_ind + 1, 0] - xy_t0[y_max_ind, 0])  # perhaps should be zero bcs xy_tp already includes all v that is needed?
		# v_p = 1

		xy_proj = np.zeros(shape=(num_p, 2))
		xy_proj[:, 0] = np.linspace(0, 1, num=num_p)  # DEFAULT VALUES COMPULSORY
		xy_proj[:, 1] = np.linspace(0, 1, num=num_p)

		'''
		THETA
		pi = bug, 2 pi = bug, 0.5 pi = straight up, 0.25 pi = 45 deg, 0.4 pi = more up, 0.1 pi = more horiz. 0.5-1 = neg x values
		Flipping doesn't change any here. 
		'''
		# theta_p = 0.25 * np.pi  # obs flipped? Increase to turn up
		# G = 9.8

		'''
		Alpha
		TODO: Use H: Its discretized
		'''
		alpha_UB = 1

		alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2.5, b=10, loc=0)  # HAVE TO HAVE A PLACEHOLDER
		alpha_mask_t = min_max_normalize_array(alpha_mask_t, y_range=[0, alpha_UB])

		# if h > 2.5:
		# 	alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2, b=2, loc=0)
		# elif stn > 2:
		# 	alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2.5, b=3, loc=0)
		# elif stn > 1.5:
		# 	alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2.5, b=6, loc=0)
		# elif stn > 1:
		# 	alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2.5, b=10, loc=0)
		# else:
		# 	alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2.5, b=10, loc=0)  # ONLY FIRST PART
		# 	adf = 5

		'''
		UPDATE: THIS EQ IS COMPLEX AND NOT WORKING. SHOULD BE EQUAL FOR ALL PARTICLES
		Perhaps need a map of the zx -> y surface and then one can know exactly where a particle will launch up from
		UPDATE2: H is now discrete!
		'''

		# if h < 2 and h >= 0.001:  # build up
		if h == 1:  # build up

			if y_min_ind - y_max_ind > 20 and x_min_ind - x_max_ind > 15 and \
					y_fall_dist > 0 and x_right_dist > 0:  # y_min occurs after y_max and x_min occurs after x_max
				x_right_dist *= 1.5
				# xy_proj[x_max_ind:, 0] += np.linspace(start=0, stop=x_right_dist, num=len(xy_proj[x_max_ind:, 1]))
				xy_proj[:, 0] += np.linspace(start=0, stop=x_right_dist, num=len(xy_proj[:, 0]))

				'''y should not go down'''
				y_fall_dist *= 2
				xy_proj[:, 1] += np.linspace(start=0, stop=y_fall_dist, num=len(xy_proj[:, 0]))

			alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=3, b=8, loc=0)  # ONLY FIRST PART
			# if random.random() < 0.05:  # flying
			# 	xy_proj[:, 0] = np.linspace(0, -150, num=num_p)
			# 	xy_proj[:, 1] = np.linspace(0, 300, num=num_p)
			# 	alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2, b=10, loc=0)  # ONLY FIRST PART

		elif h == 2:  # breaking

			if y_min_ind - y_max_ind > 20 and x_min_ind - x_max_ind > 15 and \
					y_fall_dist > 0 and x_right_dist > 0:  # y_min occurs after y_max and x_min occurs after x_max
				'''TODO: the shifting needs to correspond to the gerstner wave'''

				# if x_right_dist > 0:
				# x_right_dist += random.randint(0, 100)  # -220, 120
				# x_right_dist *= 1.5
				x_right_dist *= abs(np.random.normal(loc=1.5, scale=0.5))
				# xy_proj[x_max_ind:, 0] += np.linspace(start=0, stop=x_right_dist, num=len(xy_proj[x_max_ind:, 1]))
				xy_proj[:, 0] += np.linspace(start=0, stop=x_right_dist, num=len(xy_proj[:, 0]))

				# y_fall_dist += random.randint(300, 301)  # its flipped below
				y_fall_dist *= 1  # its flipped below
				y_fall_dist *= abs(np.random.normal(loc=1, scale=0.5))
				'''y_up_dist is all the way. But maybe it shouldnt be pushed all the way down'''
				# xy_proj[y_min_ind:, 1] = np.linspace(start=0, stop=-y_fall_dist, num=len(xy_proj[y_min_ind:, 1]))
				xy_proj[:, 1] = np.linspace(start=0, stop=-y_fall_dist, num=len(xy_proj[:, 1]))



				# num_first = len(xy_proj[:y_min_ind, 0])
				# alpha_mask_0 = beta.pdf(x=np.linspace(0, 1, num=num_first), a=2, b=2, loc=0)
				# alpha_mask_0 = min_max_normalize_array(alpha_mask_0, y_range=[0.0, alpha_UB])
				#
				# num_second = len(xy_proj[y_min_ind:, 0])
				# alpha_mask_1 = beta.pdf(x=np.linspace(0, 1, num=num_second), a=2, b=2, loc=0)
				# alpha_mask_1 = min_max_normalize_array(alpha_mask_1, y_range=[0.0, alpha_UB])

				alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2, b=6, loc=0)  # HAVE TO HAVE A PLACEHOLDER
				alpha_mask_t = min_max_normalize_array(alpha_mask_t, y_range=[0.0, alpha_UB])
				# print("Adsfasdf")

				# alpha_mask_t[0:num_first] = alpha_mask_0
				# alpha_mask_t[num_first:] = alpha_mask_1

				aa = 67

		elif h == 0:
			'''
			ChaosTK: 
			
			'''

			# if random.random() < 0.1:  # moves down
			# 	x_stop = random.randint(90, 200)
			# 	y_stop = random.randint(-20, -19)
			# else:  # moves up
			# 	x_stop = random.randint(90, 200)
			# 	y_stop = random.randint(-20, 200)

			x_stop = random.randint(200, 600)
			# # x_stop = 600
			y_stop = random.randint(-99, 50)
			# y_stop = -100
			#
			xy_proj[:, 0] = np.linspace(50, x_stop, num=num_p)
			xy_proj[:, 1] = np.linspace(-100, y_stop, num=num_p)

			alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=4, b=2, loc=0)
		else:
			raise Exception("h not 0, 1, 2")

		# alpha_mask_t = min_max_normalization(alpha_mask_t, y_range=[0.0, alpha_UB])
		alpha_mask_t = min_max_normalize_array(alpha_mask_t, y_range=[0.0, alpha_UB])
		alphas[peak_ind0:peak_ind1] = alpha_mask_t
		# TODO: ADD TEST for nan

		'''OBBBBBBSSSS REMEMBER!!!! YOUR SHIFTING IT!!!! NOT SETTING'''
		xy_t[peak_ind0:peak_ind1, :] += xy_proj

	if np.max(alphas) > 1.000:
		asdf = 5

	return xy_t, alphas, rotation0, scale


# foam_f old
# def foam_f(o1):
# 	"""
# 	New idea: Everything between start and start + num is available.
# 	So use everything and then just move object to next peak by shift.
# 	S H I F T   of static. Makes sense: If wave is crazy, foam is also crazy
# 	"""
#
# 	EARLINESS_SHIFT = 5
# 	MIN_DIST_FRAMES_BET_WAVES = 15
#
# 	xy_t = np.copy(o1.xy_t)
# 	xy_t0 = np.copy(o1.xy_t0)
# 	# dxy0 = np.copy(o1.dxy0)
#
# 	# rotation0 = np.copy(o1.dxy0[:, 1])
#
# 	rotation0 = np.full((len(o1.xy)), fill_value=-0.0001)  # CALCULATED HERE
# 	alphas = np.full(shape=(len(xy_t),), fill_value=0.0)
#
# 	'''
# 	Peaks found using xy_t
# 	But v and h found using xy_t0
# 	'''
# 	# peak_inds = scipy.signal.find_peaks(xy_t[:, 1], height=20, distance=50)[0]  # OBS 20 needs tuning!!!
# 	# peak_inds = scipy.signal.find_peaks(xy_t[:, 1], height=15, distance=10)[0]  # OBS 20 needs tuning!!!
# 	peak_inds = scipy.signal.find_peaks(xy_t[:, 1], distance=MIN_DIST_FRAMES_BET_WAVES)[0]  # OBS 20 needs tuning!!!
# 	# peak_inds = scipy.signal.find_peaks(xy_t[:, 1])[0]  # OBS 20 needs tuning!!!
# 	peak_inds -= EARLINESS_SHIFT  # neg mean that they will start before the actual peak
# 	neg_inds = np.where(peak_inds < 0)[0]
# 	if len(neg_inds) > 0:  # THIS IS NEEDED DUE TO peak_inds -= 10
# 		peak_inds = peak_inds[neg_inds[-1] + 1:]  # dont use neg inds
#
# 	'''
# 	Need to increase v with x and z.
# 	Wave breaks
# 	Use o1 id
#
# 	NEW: Now that we have peaks, we can go back to using t instead of t0
# 	Conjecture: Have to pick EITHER tp OR tp0 below.
# 	'''
#
# 	v_mult = o1.o0.gi.vmult_zx[o1.z_key, o1.x_key]
# 	h_mult = o1.o0.gi.stns_zx0[o1.z_key, o1.x_key]
# 	stn = o1.o0.gi.stns_zx0[o1.z_key, o1.x_key]
# 	h = o1.o0.gi.H[o1.z_key, o1.x_key]
# 	# h = 5
# 	x_displ = 500
#
# 	for i in range(len(peak_inds) - 1):
#
# 		peak_ind0 = peak_inds[i]
# 		peak_ind1 = peak_inds[i + 1]
#
# 		# peak_inds_x = scipy.signal.find_peaks(xy_t[peak_ind0:peak_ind1, 1])[0]
#
# 		# mid_ind = int(peak_ind1 - peak_ind0)
#
# 		'''OBS THIS WRITES TO xy_t STRAIGHT'''
# 		xy_tp = np.copy(xy_t[peak_ind0:peak_ind1])  # xy_tp: xy coords time between peaks
# 		xy_tp0 = np.copy(xy_t0[peak_ind0:peak_ind1])  # xy_tp: xy coords time between peaks
#
# 		if len(xy_tp0) < MIN_DIST_FRAMES_BET_WAVES:
# 			raise Exception("W   T   F")
#
# 		# rotation_tp = np.sin(np.linspace(-1 * np.pi, 1 *  np.pi, num=int(peak_ind1 - peak_ind0)))   # have to work with this from y_min
# 		# rotation_tp0 = np.sin(np.linspace(-0.001, -0.002 * np.pi, num=int(peak_ind1 - peak_ind0)))
# 		# rotation_tp0 = np.sin(np.linspace(-0.001, -0.6, num=int(peak_ind1 - peak_ind0)))
# 		rotation_tp = np.linspace(0, -1 * np.pi, num=int(peak_ind1 - peak_ind0))
#
# 		rotation0[peak_ind0:peak_ind1] = rotation_tp
#
# 		# rotation0[peak_ind0:peak_ind1] = np.full(shape=(int(peak_ind1 - peak_ind0),), fill_value=0.1)
#
# 		'''
# 		Generating the break motion by scaling up the Gersner rotation
# 		Might also need to shift it. Which is fine if alpha used correctly
#
# 		New thing: Instead of multiplying Gerstner circle with constant,
# 		its much cleaner to extract v at top of wave and then generating a projectile motion.
# 		BUT, this only works for downward motion
# 		'''
#
# 		# y_max_ind = int(len(xy_tp0) * 0.1)
# 		y_max_ind = EARLINESS_SHIFT
# 		y_max = xy_tp[y_max_ind, 1]
# 		y_min_ind = np.argmin(xy_tp[:, 1])
# 		y_min = xy_tp[y_min_ind, 1]
# 		y_peak1 = xy_tp[-1, 1]
#
# 		x_max_ind = np.argmax(xy_tp[:, 0])  # DOESNT WORK WITH MULTIPLE WAVES. TODO: USE PI INSTEAD
# 		x_max = xy_tp[x_max_ind, 0]
# 		x_min_ind = np.argmin(xy_tp[:, 0])
# 		x_min = xy_tp[x_min_ind, 0]
# 		x_peak_ind1 = xy_tp[-1, 0]
#
# 		'''This can happen if peak thresholds are too large'''
# 		# if y_max_ind >= y_min_ind:
# 		# 	raise Exception("adfasdf")
# 		# assert(y_max_ind < y_min_ind)
# 		# if y_max_ind >= x_max_ind:
# 		# 	print("ASDfasdfasdfasdf")
# 		# 	continue
# 		# raise Exception("Asdfadfffff")
# 		# assert(y_max_ind < x_max_ind)
#
# 		# start_x = xy_tp0[y_max_ind, 0]
# 		# start_y = xy_tp0[y_max_ind, 1]
#
# 		# start_x = xy_tp[y_max_ind, 0]
# 		# start_y = xy_tp[y_max_ind, 1]
#
# 		'''
# 		NUM HERE IS FOR PROJ. STARTS WHEN Y AT MAX
# 		NUM SHOULD BE SPLIT INTO TWO PARTS (Maybe not)
# 		NUM_P IS ONLY PROJ
# 		NUM_B IS FOR RISING		'''
#
# 		# num_p = len(xy_tp0[y_max_ind:x_max_ind])  # OBS! This is where proj motion is used
# 		# num_p = len(xy_tp0[y_max_ind:])  # OBS! This is where proj motion is used
# 		num_p = len(xy_tp0)
# 		# num_b = len(xy_tp0[x_max_ind:])
#
# 		''''''
# 		v_frame = abs(xy_t0[y_max_ind + 1, 0] - xy_t0[
# 			y_max_ind, 0])  # perhaps should be zero bcs xy_tp already includes all v that is needed?
# 		v_p = 1
# 		# v_b = 80
# 		# v_frame = abs(xy_t0[y_max_ind + 1, 0] - xy_t0[y_max_ind, 0])   # perhaps should be zero bcs xy_tp already includes all v that is needed?
# 		# v_frame *= 2
# 		# v_frame *= v_mult
#
# 		xy_proj = np.zeros(shape=(num_p, 2))
# 		# xy_b = np.zeros(shape=(num_b, 2))
# 		# xy_b[:, 0] = v_frame
#
# 		'''
# 		THETA
# 		pi = bug, 2 pi = bug, 0.5 pi = straight up, 0.25 pi = 45 deg, 0.4 pi = more up, 0.1 pi = more horiz. 0.5-1 = neg x values
# 		Flipping doesn't change any here.
# 		'''
# 		theta_p = 0.25 * np.pi  # obs flipped? Increase to turn up
# 		# theta_b = 0.2  # np.random.normal(loc=0.25, scale=0.1) * np.pi
# 		# theta_b_flip_y = 1#np.random.choice([-1, 1], p=[0.2, 0.8])
# 		G = 9.8
#
# 		'''
# 		xy_tp0 gives all highs and lows from stns!
# 		if h_mult is to be used it should not use stns
# 		OBS this h is ADDED to h that is already in Gerstner. So needs to be small
# 		'''
# 		# h = xy_tp0[y_max_ind, 1] - xy_tp0[x_max_ind, 1]  # OBS HACK BELOW more, = more fall ALSO TO RIGHT
# 		# h = 10 * stn  # OBS HACK BELOW more, = more fall ALSO TO RIGHT
#
# 		'''
# 		UPDATE: THIS EQ IS COMPLEX AND NOT WORKING. SHOULD BE EQUAL FOR ALL PARTICLES
# 		Perhaps need a map of the zx -> y surface and then one can know exactly where a particle will launch up from
# 		'''
# 		# t_flight_p = (v_p * np.sin(theta_p) + np.sqrt((v_p * np.sin(theta_p)) ** 2 + 2 * G * h)) / G  # frames?
# 		# t_flight_p = 10
# 		# t_flight_p = (np.sqrt(2 * G * h)) / G
# 		# t_flight_b = (v_b * np.sin(theta_b) + np.sqrt((v_b * np.sin(theta_b)) ** 2 + 2 * G * h)) / G
# 		# t_flight_b = 3 * v_b * np.sin(theta_b) / G
#
# 		# t_lin_b = np.linspace(0, t_flight_b, num_)
#
# 		# xy_proj[:, 0] = v_p * np.cos(theta_p) * t_lin_p  # THIS IS LINEAR!!!!
#
# 		# xy_proj[:, 0] = np.linspace(0, x_displ, num=num_p)
# 		# xy_proj[:, 1] = v_p * np.sin(theta_p) * 2 * t_lin_p - 0.5 * G * t_lin_p ** 2
# 		# t_lin_p = np.linspace(0, 0.0001, num_p)  # the more this is increased, the more fall
#
# 		if h < 1.6 and h >= 0.001:
# 			xy_proj[:, 0] = np.linspace(0, 400, num=num_p)
# 		# xy_proj[:, 1] = 15 * t_lin_p
# 		elif h > 1.6:
# 			xy_proj[:, 0] = np.linspace(0, 400, num=num_p)  # 1000
# 			# xy_proj[:, 1] = 4 * t_lin_p - t_lin_p ** 2  # first one: more=more v up, i.e. will fall less
# 			if y_min_ind - y_max_ind > 20:  # y_min occurs after y_max
# 				y_up_dist = y_peak1 - y_min
# 				if y_up_dist > 50:
# 					y_up_dist += random.randint(-20, 20)
# 					'''y_up_dist is all the way. But maybe it shouldnt be pushed all the way down'''
# 					xy_proj[y_min_ind:, 1] = np.linspace(start=0, stop=-0.5 * y_up_dist,
# 					                                     num=len(xy_proj[y_min_ind:, 1]))
# 					aa = 5
#
# 			if x_min_ind - x_max_ind > 20:  # x_min occurs after x_max
# 				x_left_dist = x_max - x_min
# 				if x_left_dist > 50:
# 					x_left_dist += random.randint(-20, 20)
# 					xy_proj[x_max_ind:, 0] += np.linspace(start=0, stop=x_left_dist, num=len(xy_proj[x_max_ind:, 1]))
# 					aa = 5
# 		else:
# 			xy_proj[:, 0] = np.linspace(0, 200, num=num_p)
# 		# xy_proj[:, 1] = 30 * t_lin_p
#
# 		# xy_b[:, 0] = v_b * np.cos(theta_b) * t_lin_b
# 		# xy_b[:, 1] = theta_b_flip_y * v_b * np.sin(theta_b) * 2 * t_lin_b - 0.5 * G * t_lin_b ** 2
#
# 		# xy_b += 1
# 		# xy_b[:, 0] *= 35
# 		# xy_b[:, 1] *= 1
# 		# xy_b[:, 1] = t_flight = 4 * gi['v'] * np.sin(gi['theta']) / G
#
# 		# xy_proj[:, 0] += start_x
# 		# xy_proj[:, 1] += start_y
#
# 		# xy_tp0[y_max_ind:, :] = xy_proj  # Old. Setting is more difficult than shifting?
# 		# xy_t0[y_max_ind:y_max_ind + num, :] += xy_proj  # BUG: Cant use y_max_ind HERE!!!
# 		# xy_t0[peak_ind0 + y_max_ind:peak_ind0 + y_max_ind + num_p, :] += xy_proj  # NEW: Its a shift. Conceptually easier than a reset.
# 		# xy_t0[peak_ind0 + y_max_ind:peak_ind0 + x_max_ind, :] += xy_proj
# 		# xy_t0[peak_ind0 + y_max_ind:peak_ind1, :] += xy_proj
#
# 		'''OBBBBBBSSSS REMEMBER!!!! YOUR SHIFTING IT!!!! NOT SETTING'''
# 		xy_t0[peak_ind0:peak_ind1, :] += xy_proj
#
# 		# start_xy_b = xy_t0[peak_ind0 + x_max_ind - 1, :]
# 		# xy_b += start_xy_b
# 		# xy_t0[peak_ind0 + y_min_ind:peak_ind1, 1] += xy_b[:, 1]
# 		# xy_t0[peak_ind0 + x_max_ind:peak_ind1, :] = xy_b
# 		# xy_t0[peak_ind0 + x_max_ind:peak_ind1, 1] *= -1  # madness
#
# 		'''aaa'''
# 		alpha_UB = 1
# 		if stn > 2.5:
# 			alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2, b=2, loc=0)
# 		elif stn > 2:
# 			alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2.5, b=3, loc=0)
# 		elif stn > 1.5:
# 			alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2.5, b=6, loc=0)
# 		elif stn > 1:
# 			alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2.5, b=9, loc=0)
# 		else:
# 			alpha_mask_t = beta.pdf(x=np.linspace(0, 1, len(xy_tp0)), a=2.5, b=10, loc=0)  # ONLY FIRST PART
# 			adf = 5
# 		alpha_mask_t = min_max_normalization(alpha_mask_t, y_range=[0.0, alpha_UB])
#
# 		alphas[peak_ind0:peak_ind1] = alpha_mask_t
#
# 	return xy_t0, alphas, rotation0

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
