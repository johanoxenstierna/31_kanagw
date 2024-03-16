# PROJECTILES

import copy

# from sh_info.shInfoAbstract import ShInfoAbstract
import P as P
import random
import numpy as np
from scipy import stats

class Projectiles_info:
    """
    These are the parameters of O0 and children.
    Basically, the separation between this and finish_info, is that the parameters in most cases
    arent enough to describe the full motion. So everything that is more akin to finishing the
    motion is in finish_info, and everything that is more like an input to finish_info
    is done here. NOTE: These are unique types of objects, so finish_info will need if-else
    whenever the unique stuff isnt sorted out here.
    """

    def __init__(_s):

        _s.id = 'projectiles'  # PROJECTILES
        # _s.extent = "static"
        _s.frame_ss = [0, P.FRAMES_STOP - 50]
        _s.zorder = 95

        # _s.child_names = ['O1']  # This is used by gen_layers later to load correct pics

        # o1_gi['down_offsets'] = np.linspace(0, 200, num=P.NUM_O1_WAVES)
        _s.o1_gi = _s.gen_o1_gi()  # OBS: sp_gi generated in f class. There is no info class for f.
        _s.o2_gi = _s.gen_o2_gi()

        '''Below are distributed among different children instances'''
        # _s.o1_init_frames = [5, 10, 15, 20, 25, 60, 80, 100, 120, 140, 150, 180, 220]
        _s.o1_init_frames = [5, 10, 15] + list(np.random.random_integers(low=5, high=1400, size=P.NUM_O1_PROJS * 5 * 3))  # OBS MUTABLE. OBS see o1  # 5 is number of init framer per o, 3 is num pics
        _s.o1_down_offsets = np.linspace(0, 50, num=P.NUM_O1_PROJS)

    def gen_o1_gi(_s):
        """
        This has to be provided because the fs are generated w.r.t. sh.
        This is like the constructor input for F class
        """
        # FRAMES_TOT = 200  # MUST BE HIGHTER THAN SP.FRAMES_TOT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        NUM_RANDS = max(P.NUM_PROJ_O2_PER_O1, P.NUM_WAVE_O2_PER_O1) + 5

        o1_gi = {
            'init_frames': None,  # New: done later
            'frames_tot': 500,  # only used for init
            'frame_expl': np.random.randint(low=180, high=199, size=1)[0],
            'scale_ss': [0.01, 1.1],
            'frame_ss': None,  # simpler with this
            'ld': [300, 600],
            'left_offsets': None,  # BELOW [-500, 500] num doesnt matter cuz pos = random.randint(0, 20)
            'down_offsets': None,  # BELOW
            'zorder': 50
        }

        o1_gi['left_offsets'] = np.zeros(shape=(NUM_RANDS,))  # PER O2

        return o1_gi

    def gen_o2_gi(_s):
        """
        UPDATE: THESE ARE NO LONGER CHILDREN OF F,
        THEIR INIT FRAMES CAN BE SET BY F THOUGH.
        """
        sps_gi = {
            # 'alpha_y_range': [0.5, 0.9],
            'init_frames': None,  # ONLY FOR THIS TYPE
            'frames_tot': 300,  # MUST BE LOWER THAN SP.FRAMES_TOT. MAYBE NOT. INVOLVED IN BUG  OBS
            'v_loc': 20, 'v_scale': 5,  # 50 THIS IS HOW HIGH THEY GO (not how far down)
            # 'theta_scale': 0.0,  #
            'sp_len_start_loc': 1, 'sp_len_start_scale': 1,
            'sp_len_stop_loc': 2, 'sp_len_stop_scale': 1,  # this only cov  ers uneven terrain
            'special': False,
            'ld_init': [None, None],  # set by f
            'ld': [None, None],  # set by f
            'ld_offset_loc': [0, 0],  # NEW: Assigned when inited
            'ld_offset_scale': [0, 0],  # [125, 5]
            'rgb_start': [0.1, 0.2],  #
            'up_down': 'up',
            'out_screen': False,
            'zorder': 1000
        }

        return sps_gi
