"""GERSTNER"""

import copy

# from sh_info.shInfoAbstract import ShInfoAbstract
# import scipy.stats
from scipy.stats import beta, gamma
import scipy
from src.trig_functions import min_max_normalization

import P as P
import random
import numpy as np


class Waves_info:
    """
    The class instance itself is the container for all the info,
    for the parent o0 they are
    """

    def __init__(_s):

        _s.id = 'waves'  # Gerstner
        _s.frame_ss = [0, P.FRAMES_STOP - 50]
        _s.zorder = None

        _s.o1_init_frames = [1]  # ALWAYS
        '''
        left_z is the SHEAR. So points are shifted more to right the closer to screen they are.
        Perhaps only used to reduce number of points. 
        This means that direction vector d needs to be tuned TOGETHER with it. 
        This shear can probably be removed if the image from which point pngs are taken is sheared instead. 
        '''
        # _s.o1_left_x = np.linspace(0, 1200, num=P.NUM_X)  # this is per 'a' 'b', i.e. horizontal
        _s.o1_left_x = np.linspace(-100, 1200, num=P.NUM_X)  # this is per 'a' 'b', i.e. horizontal
        _s.o1_left_z = np.linspace(0, 0, num=P.NUM_Z)  # 200, 0 this is per z i.e. away from screen. SHEAR. Its only used to reduce number of points
        # _s.o1_down_z = np.linspace(-50, 100, num=P.NUM_Z)
        _s.o1_down_z = np.linspace(-50, 200, num=P.NUM_Z)  # 40, 200 first one is starting above lowest
        # _s.o1_steepnessess_z = np.linspace(0.9, 0.9, num=P.NUM_Z)  # OBS ONLY BETWEEN 0 and 1. OBS NO STEEPNESS X
        # _s.o1_steepnessess_x = np.linspace(0.3, 1, num=P.NUM_X)  # OBS ONLY BETWEEN 0 and 1. OBS NO STEEPNESS X
        # _s.stns_x = np.linspace(1.5, 0.3, num=P.NUM_X)  # OBS ONLY BETWEEN 0 and 1. OBS NO STEEPNESS X
        # _s.stns_x = np.geomspace(start=2, stop=0.1)

        '''TODO: THESE SHOULD BE BETA DISTS'''
        _s.stns_zx0 = np.zeros(shape=(P.NUM_Z, P.NUM_X))
        _s.stns_zx1 = np.zeros(shape=(P.NUM_Z, P.NUM_X))

        stns_z0 = beta.pdf(x=np.linspace(0, 1, P.NUM_Z), a=3, b=5, loc=0)  # a>b BIGGEST FURTHEST AWAY
        # w0 = beta.pdf(x=np.linspace(0, 1, P.NUM_Z), a=3, b=3, loc=0)
        # w1 = beta.pdf(x=np.linspace(0, 1, P.NUM_Z), a=6, b=2, loc=0)
        # aa = w0 + w1
        stns_z0 = min_max_normalization(stns_z0, y_range=[0.7, 1.5])  # OBS BIGGEST IND IS FURTEST FROM SCREEN
        peak = scipy.signal.find_peaks(stns_z0)[0][-1]
        stns_z0[peak:] *= np.exp(np.linspace(start=0, stop=-5, num=P.NUM_Z - peak))

        w0 = beta.pdf(x=np.linspace(0, 1, P.NUM_X), a=4, b=20, loc=0)
        w1 = beta.pdf(x=np.linspace(0, 1, P.NUM_X), a=20, b=20, loc=0)
        w2 = beta.pdf(x=np.linspace(0, 1, P.NUM_X), a=20, b=4, loc=0)
        # stns_x0 = beta.pdf(x=np.linspace(0, 1, P.NUM_X), a=5, b=5, loc=0)
        stns_x0 = min_max_normalization(w0 + w1 + w2, y_range=[0.5, 2.5])
        # peak = scipy.signal.find_peaks(stns_x0)[0][-1]
        # stns_x0[peak:] *= np.exp(np.linspace(start=0, stop=-5, num=P.NUM_X - peak))

        stns_z1 = beta.pdf(x=np.linspace(0, 1, P.NUM_Z), a=5, b=2, loc=0)  # a>b BIGGEST FURTHEST AWAY
        stns_z1 = min_max_normalization(stns_z1, y_range=[0.2, 1.5])  # OBS BIGGEST IND IS FURTEST FROM SCREEN
        peak = scipy.signal.find_peaks(stns_z1)[0][0]
        stns_z1[peak:] *= np.exp(np.linspace(start=0, stop=-5, num=P.NUM_Z - peak))

        stns_x1 = beta.pdf(x=np.linspace(0, 1, P.NUM_X), a=4, b=5, loc=0)
        stns_x1 = min_max_normalization(stns_x1, y_range=[0.2, 1.5])
        peak = scipy.signal.find_peaks(stns_x1)[0][0]
        stns_x1[peak:] = np.exp(np.linspace(start=0, stop=-5, num=P.NUM_X - peak))

        for i in range(P.NUM_Z):
            for j in range(P.NUM_X):
                stn_z = stns_z0[i]
                stn_x = stns_x0[j]
                stn_zx = 0.2 * stn_z + 0.8 * stn_x
                _s.stns_zx0[i, j] = stn_zx  # OBS BIGGEST ROW IS FURTEST FROM SCREEN

                stn_z = stns_z1[i]
                stn_x = stns_x1[j]
                stn_zx = 0.2 * stn_z + 0.8 * stn_x
                _s.stns_zx1[i, j] = stn_zx  # OBS BIGGEST ROW IS FURTEST FROM SCREEN

        '''Distance_mult applied after static built with  gerstner(). Then b and f built on that.  '''
        _s.distance_mult = np.linspace(1, 0.8, num=P.NUM_Z)  # DECREASES WITH ROWS  # NO HORIZON WITHOUT THIS

        # NEEDS TO BE ALIGNED WITH X TOO
        _s.o1_left_starts_z = np.linspace(0.0000, 0.0001, num=P.NUM_Z)  # highest vs lowest one period diff

        _s.o1_gi = _s.gen_o1_gi()
        # _s.o2_gi = _s.gen_o2_gi()

    def gen_o1_gi(_s):
        """
        This has to be provided because the fs are generated w.r.t. sh.
        This is like the constructor input for F class
        """

        o1_gi = {
            'init_frames': None,
            'frames_tot': P.FRAMES_STOP - 25,
            'frame_ss': None,
            'ld': [None, None],  # x z !!!
            'left_offsets': None,
            'zorder': 5
        }

        '''OFFSETS FOR O2
        THIS GIVES LD FOR O2!!!
        '''
        # o1_gi['left_offsets'] = np.linspace(-400, -0, num=P.NUM_X)  # USED PER 02

        return o1_gi

    # def gen_o2_gi(_s):
    #     """
    #     UPDATE: THESE ARE NO LONGER CHILDREN OF F,
    #     THEIR INIT FRAMES CAN BE SET BY F THOUGH.
    #     """
    #     o2_gi = {
    #         'alpha_y_range': [1, 1],
    #         'init_frames': None,  # ONLY FOR THIS TYPE
    #         'frames_tot': 1200,  # MUST BE LOWER THAN SP.FRAMES_TOT. MAYBE NOT. INVOLVED IN BUG  OBS
    #         'v_loc': 50, 'v_scale': 4,  # 50 THIS IS HOW HIGH THEY GO (not how far down)
    #         'ld': [None, None],  # left-down
    #         'up_down': 'up',
    #         'zorder': 1000
    #     }
    #
    #     return o2_gi
