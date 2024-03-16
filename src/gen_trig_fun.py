
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from src.trig_functions import _normal, _sigmoid, _gamma, _log, _log_and_linear, min_max_normalization

# def gen_alpha(g_obj, frames_tot=None, y_range=None, plot=False):
#
# 	if frames_tot == None:
# 		X = np.arange(0, g_obj.gi['frame_ss'][-1] - g_obj.gi['frame_ss'][0])
# 	else:
# 		X = np.arange(0, frames_tot)
#
# 	'''NEW: ADD DIST_TO_MIDDLE AND THETA'''
# 	alpha1 = np.asarray(
# 		([_sigmoid(x, grad_magn_inv=len(X) / 8, x_shift=3, y_magn=1., y_shift=0) for x in X]))
#
# 	alpha2 = np.sin((X - 20) / 16) / 4
#
# 	alpha = alpha1 + alpha2
#
# 	alpha = min_max_normalization(alpha, y_range=y_range)
#
# 	return alpha

def gen_alpha(o, _type):

	# X = np.arange(0, o.gi['frames_tot'])

	if _type == 'o1_projectiles':
		alphas = np.ones(shape=(o.gi['frames_tot']))
		alphas[o.gi['frame_expl'] + 5:] = 0.1

	elif _type == 'o2_projectiles':
		# alphas = np.ones(shape=(o.gi['frames_tot']))

		X = np.arange(0, o.gi['frames_tot'])
		alphas = np.asarray(([_sigmoid(x, grad_magn_inv=- len(X) / 8, x_shift=-8, y_magn=1., y_shift=0) for x in X]))
		alphas = min_max_normalization(alphas, y_range=[0, 1])

	elif _type == 'o1_clouds':
		# alphas = np.ones(shape=(o.gi['frames_tot']))

		X = np.arange(0, o.gi['frames_tot'])
		alphas = _normal(X, mean=o.gi['frames_tot'] / 2, var=o.gi['frames_tot'] / 3, y_range=[0, 0.2])  # 720
		asd = 5

	return alphas



