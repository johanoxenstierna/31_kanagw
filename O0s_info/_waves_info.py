"""GERSTNER"""

import copy

# from sh_info.shInfoAbstract import ShInfoAbstract
# import scipy.stats
from scipy.stats import beta, gamma
import scipy
from src.trig_functions import min_max_normalization
from O0s_info._waves_helper import *

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

        _s.o1_left_x = np.linspace(-100, 1200, num=P.NUM_X)  # this is per 'a' 'b', i.e. horizontal
        _s.o1_left_z = np.linspace(0, 0, num=P.NUM_Z)  # 200, 0 this is per z i.e. away from screen. SHEAR. Its only used to reduce number of points

        _s.o1_down_z = np.linspace(300, 301, num=P.NUM_Z)  # 40, 200 first one is starting above lowest

        if P.COMPLEXITY == 1:
            _s.o1_down_z = np.linspace(50, 200, num=P.NUM_Z)  # 40, 200 first one is starting above lowest

        # _s.stns_ZX, _s.H = gen_stns_old()
        if P.QUERY_STNS == 0:
            print("Building stns ...")
            _s.stns_TZX, _s.TH = gen_stns()
            print("Done uilding stns ...")
        else:
            _s.stns_TZX = np.load('./O0s_info/stns_TZX.npy')
            _s.TH = np.load('./O0s_info/TH.npy')


        '''Distance_mult applied after static built with  gerstner(). Then b and f built on that.  
        TODO: stns_zx0 should be tilted
        '''
        _s.distance_mult = np.linspace(1, 0.8, num=P.NUM_Z)  # DECREASES WITH ROWS  # NO HORIZON WITHOUT THIS
        # _s.h_mult = np.geomspace(1, 0.1, num=P.NUM_Z)

        ff = 5
        # _s.H[:, int(P.NUM_X / 2):] = 0
        # _s.stns_zx0[:, int(P.NUM_X / 2):] /= 1.2
        # _s.stns_ZX[0, :, int(P.NUM_X / 2):] /= 1.2
        asdf = 4
        '''
        OBS MAKING H SMALL IS ALSO SLOWING X MOVEMENT
        UPDATE: BASING H_MULT ON STNS DOESNT MAKE SENSE SINCE F USE STNS ANYWAY
        '''
        # _s.h_mult = np.copy(_s.stns_zx0) * 0.2
        # aa = np.linspace(start=1, stop=0)

        # NEEDS TO BE ALIGNED WITH X TOO
        '''This is probably depr. Wave needs to break at left first, but below was used 
         to fix init_frame prob.'''
        _s.o1_left_starts_z = np.linspace(0.0000, 0.0001, num=P.NUM_Z)  # highest vs lowest one period diff

        Z, X = np.mgrid[2:0.5:complex(0, P.NUM_Z), 0.5:2:complex(0, P.NUM_X)]
        _s.vmult_zx = 0.5 * Z + 0.5 * X
        # TODO: zx for pic scales

        _s.o1_gi = _s.gen_o1_gi()

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

