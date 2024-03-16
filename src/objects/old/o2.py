"""spark"""

from src.gen_extent_triangles import *
from src.objects.abstract import AbstractObject, AbstractSSS
import P as P
import numpy as np
from copy import deepcopy
import random
from src.projectile_functions import *
from src.wave_funcs import gerstner_wave
from src.gen_trig_fun import gen_alpha, min_max_normalization
from src.trig_functions import _sigmoid
import matplotlib as mpl
from scipy.stats import beta

class O2C(AbstractObject, AbstractSSS):

    def __init__(_s, o0, o2_id_int, o1):
        AbstractObject.__init__(_s)

        _s.o0 = o0
        _s.id = o0.id + "_" + o1.id + "_" + str(o2_id_int)

        _s.o1 = o1
        _s.gi = deepcopy(o0.gi.o2_gi)  #
        _s.o2_id_int = o2_id_int
        # _s.sp_lens = None
        # _s.ars_bool = 0

        AbstractSSS.__init__(_s, o0, _s.id)

    def gen_scale_vector(_s):

        scale_ss = []
        return scale_ss

    def dyn_gen(_s, i, gi=None):

        """
        Basically everything moved from init to here.
        This can only be called when init frames are synced between
        """

        def fun():
            pass

        _s.finish_info()
        _s.init_frame = _s.set_init_frame(i)

        # _s.set_frames_tot()  # SAME
        if _s.o0.id == 'projectiles':
            _s.xy_t = simple_projectile(gi=_s.gi)  # OUTPUTS BOTH LEFT (NEG) AND RIGHT (POS)
            _s.xy_t = flip_projectile_x(_s)
            # _s.xy = shift_projectile(_s.xy_t, origin=(_s.gi['ld'][0], _s.gi['ld'][1]), gi=_s.gi)
            _s.xy = flip_and_shift_object(_s.xy_t, origin=(_s.o1.XY[_s.o1.gi['frame_expl'], 0],
                                                      _s.o1.XY[_s.o1.gi['frame_expl'], 1]), gi=_s.gi)
            _s.alphas = gen_alpha(_s, _type='o2_projectiles')
        elif _s.o0.id == 'waves':
            _s.xy_t, _s.alphas = gerstner_wave(gi=_s.gi)
            # _s.xy = _s.xy_t
            _s.xy = flip_and_shift_object(_s.xy_t, origin=(_s.gi['ld'][0], _s.gi['ld'][1]), gi=_s.gi)
            # _s.alphas = np.zeros(shape=(len(_s.xy)))

        # _s.sp_lens = _s.set_sp_lens()  # PERHAPS THETA INSTEAD?

        """HERE SET ALPHA BASED ON THETA AND DIST TO MIDDLE"""

        assert (len(_s.alphas) == len(_s.xy))

    def set_init_frame(_s, i):

        """The number i (an init frame, must be in init_frames of o1,
        thats what index(i) does."""

        index_init_frames = _s.o1.gi['init_frames'].index(i)
        init_frame_o1 = _s.o1.gi['init_frames'][index_init_frames]

        # index_init_frames = _s.o1.o0.gi.o1_init_frames.index(i)
        # init_frame_f = _s.o1.o0.gi.o1_init_frames[index_init_frames]

        # init_frame_offset = random.randint(0, _s.gi['init_frame_max_dist']) # np.random.poisson(10, 100)
        # init_frame_offset = np.random.poisson(1, 1)[0]
        # init_frame_offset = int(beta.rvs(a=2, b=5, loc=0, scale=200, size=1)[0])  # OBS
        # init_frame_offset = min(120, init_frame_offset)
        # init_frame = init_frame_f + init_frame_offset

        '''Without init frame offset'''
        if _s.o0.id == 'projectiles':
            init_frame = init_frame_o1 + _s.o1.gi['frame_expl']
        elif _s.o0.id == 'waves':
            init_frame = init_frame_o1


        # '''OBS special lateness to side ones'''
        # if P.NUM_FS == 2:
        #     pass
        # else:
        #     pass
        #     # if _s.gi['dist_to_theta_loc'] > 0.15:
        #     #     init_frame += 40

        return init_frame

    def finish_info(_s):

        """
        Modifies parameters for specific type of o2
        """

        '''FROM f. OBS f gi is used for some things and sp gi used for others'''
        _s.gi['ld'] = deepcopy(_s.o1.gi['ld'])
        _s.gi['ld'][0] += _s.o1.gi['left_offsets'][_s.o2_id_int]

        if _s.o0.id == 'waves':  # singleton
            _s.gi['steepness'] = _s.o1.gi['steepness']
            _s.gi['o1_left_start'] = _s.o1.gi['o1_left_start']

        _s.gi['v'] = abs(np.random.normal(loc=_s.gi['v_loc'], scale=_s.gi['v_scale']))

        if _s.o0.id == 'projectiles':  # singleton
            # _s.gi['theta'] = np.random.normal(loc=_s.gi['theta_loc'], scale=_s.gi['theta_scale'])
            _s.gi['theta'] = np.random.uniform(low=0, high=2 * np.pi)

            '''ld'''

            _s.gi['ld_offset'] = [0, 0]

            _s.gi['ld_init'] = [_s.gi['ld'][0] + _s.gi['ld_offset'][0], _s.gi['ld'][1] + _s.gi['ld_offset'][1]]  # BEFORE
            _s.gi['ld'] = deepcopy(_s.gi['ld_init'])  # IT GETS SHIFTED TO OPPOSITE


            '''New:'''
            start = random.uniform(_s.gi['rgb_start'][0], _s.gi['rgb_start'][1])
            # start = min(_s.gi['rgb_start'][1], start - random.random())
            end = max(0.8, random.uniform(0.8, start - 0.1))

            x = np.linspace(start, end, _s.gi['frames_tot'])  # no need to flip since it starts hot
            rgb = mpl.colormaps[_s.o1.cmap](x)[:, 0:3]  # starts as cold

            _s.R = rgb[:, 0]
            _s.G = rgb[:, 1]
            _s.B = rgb[:, 2]

    def set_sp_lens(_s):
        """y_do_shift is currently hardcoded"""

        sp_len_start = np.random.randint(low=2, high=16)

        sp_len_stop = np.random.randint(low=sp_len_start, high=sp_len_start + 2)

        sp_lens = np.linspace(sp_len_start, sp_len_stop, num=_s.gi['frames_tot'], dtype=int)

        return sp_lens







