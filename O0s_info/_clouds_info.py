

# PROJECTILES

import copy

# from sh_info.shInfoAbstract import ShInfoAbstract
import P as P
import random
import numpy as np
from scipy import stats

class Clouds_info:
    """
    These are the parameters of O0 and children.
    Basically, the separation between this and finish_info, is that the parameters in most cases
    arent enough to describe the full motion. So everything that is more akin to finishing the
    motion is in finish_info, and everything that is more like an input to finish_info
    is done here. NOTE: These are unique types of objects, so finish_info will need if-else
    whenever the unique stuff isnt sorted out here.
    """

    def __init__(_s):

        _s.id = 'clouds'  # PROJECTILES
        # _s.extent = "static"
        _s.frame_ss = [0, P.FRAMES_STOP - 50]
        _s.zorder = 95

        # _s.child_names = ['O1']  # This is used by gen_layers later to load correct pics

        # o1_gi['down_offsets'] = np.linspace(0, 200, num=P.NUM_O1_WAVES)
        _s.o1_gi = _s.gen_o1_gi()  # OBS: sp_gi generated in f class. There is no info class for f.
        _s.o2_gi = {}

        '''Below are distributed among different children instances'''
        # _s.o1_init_frames = list(np.random.random_integers(low=5, high=300, size=P.NUM_O1_CLOUDS * 5 * 4))  # OBS MUTABLE. OBS see o1
        i0 = [1] * 100  # 5=num init frames per o, 4=num o pics
        i1 = [300] * 100
        i2 = [600] * 100
        i3 = [900] * 100
        i4 = [1200] * 100

        _s.o1_init_frames = i0 + i1 + i2 + i3 + i4
        _s.o1_down_offsets = np.linspace(0, 50, num=P.NUM_O1_PROJS)

    def gen_o1_gi(_s):
        """
        This has to be provided because the fs are generated w.r.t. sh.
        This is like the constructor input for F class
        """

        o1_gi = {
            'init_frames': None,  # New: done later
            'frames_tot': 295,  # only used for init
            'scale_ss': [None, None],
            'frame_ss': None,  # simpler with this
            'ld': [400, 200],
            # 'left_mid': 640,
            # 'left_offsets': None,  # BELOW [-500, 500] num doesnt matter cuz pos = random.randint(0, 20)
            # 'down_offsets': None,  # BELOW
            # 'theta_loc': (6/10) * 2 * np.pi,  # set at init from offsets
            # 'theta_offsets': None,  # list(np.linspace(0.3, -0.3, num=NUM_RANDS)), #[0.5, -0.5],
            # 'init_frame_x_offsets': list(np.linspace(30, 0, num=NUM_RANDS - 25, dtype=int)) + list(np.linspace(0, 30, num=NUM_RANDS - 15, dtype=int)),
            # 'init_frames_dirichlet': None,
            # 'x_mov': list(np.linspace(0, -15, num=FRAMES_TOT)),  # SPECIAL
            'zorder': 50
        }

        return o1_gi
