import random

import numpy as np
from copy import deepcopy

import P as P
from src.gen_trig_fun import *
from src.objects.abstract import AbstractObject, AbstractSSS
from src.wave_funcs import *

class O1C(AbstractObject, AbstractSSS):

    def __init__(_s, o1_id, pic, o0, type):
        AbstractObject.__init__(_s)
        _s.id = o1_id

        _s.id_s = o1_id.split('_')
        _s.x_key = int(_s.id_s[0])
        _s.z_key = int(_s.id_s[1])  # biggest z_key is at the top

        z_key_g = 20000 - _s.z_key * 100  # the bigger the z_key, the smaller the zorder. Each Z: 100, each X: 1
        _s.zorder = z_key_g + _s.x_key

        _s.type = type
        _s.o0 = o0  # parent
        _s.pic = pic  # the png
        _s.centroid = [int(pic.shape[0] / 2), int(pic.shape[1] / 2)]
        # _s.type = type
        # _s.gi = deepcopy(o0.gi.o1_gi)  # OBS!  COPY SHOULD NOT BE THERE. SHOULD BE READ-ONLY.OK WHILE FEW OBJECTS.
        _s.gi = o0.gi.o1_gi  # REMOVE OBS!  COPY SHOULD NOT BE THERE. SHOULD BE READ-ONLY.OK WHILE FEW OBJECTS.
        # ONLY OBJECTS THAT ARE MUTABLE ARE TO BE COPIED

        AbstractSSS.__init__(_s, o0, o1_id)

        _s.o1_sib = None  # sibling. THIS IS THE CHECK CONDITION

        _s.O2 = {}
        _s.alphas = None

        _s.gi['init_frames'] = [_s.o0.gi.o1_init_frames[0]]  # same for all
        _s.gi['ld'][0] = _s.o0.gi.o1_left_x[_s.x_key] + _s.o0.gi.o1_left_z[_s.z_key]  # last one is shear! probably removed later
        _s.gi['ld'][1] = _s.o0.gi.o1_down_z[_s.z_key]

        if P.COMPLEXITY == 0:
            pass

        # elif P.COMPLEXITY == 1:
        # NOT HERE
        # _s.gi['ld'][0] += random.randint(-10, 10)
        # _s.gi['ld'][1] += random.randint(-5, 5)

        # _s.gi['steepness'] = _s.o0.gi.o1_steepnessess_z[_s.z_id] #+ np.random.randint(low=0, high=50, size=1)[0]
        # _s.gi['steepness'] = _s.o0.gi.stns_zx[_s.z_key, _s.x_key] #+ np.random.randint(low=0, high=50, size=1)[0]

        _s.gi['o1_left_start_z'] = _s.o0.gi.o1_left_starts_z[_s.z_key] #+ np.random.randint(low=0, high=50, size=1)[0]

        _s.set_frame_ss(0, _s.gi['frames_tot'])

    def gen_scale_vector(_s):

        scale_ss = []
        return scale_ss

    def gen_static(_s):

        """
        Basically everything moved from init to here.
        This can only be called when init frames are synced between
        TODO: TENSOR HEATMAP WITH X, Z AND Y. FROM THAT F CAN BE GENERATED
        """


        '''NEXT: add rotation here'''

        _s.xy_t, _s.dxy, \
        _s.alphas, _s.rotation, _s.peaks, \
        _s.xy_t0, _s.dxy0, \
        _s.xy_t1, _s.dxy1, \
        _s.xy_t2, _s.dxy2, _s.scale, _s.YT \
            = gerstner_waves(o1=_s, o0=_s.o0)

        # _s.gi['ld'][0] += random.randint(-10, 10)
        # _s.gi['ld'][1] += random.randint(-5, 5)
        '''shifting '''
        _s.xy = np.copy(_s.xy_t)
        _s.xy *= _s.o0.gi.distance_mult[_s.z_key]
        _s.xy[:, 0] += _s.gi['ld'][0] + _s.gi['o1_left_start_z']  # last one should be removed ev
        _s.xy[:, 1] += _s.gi['ld'][1]  # - xy[0, 1]

        # _s.scale = np.ones(shape=(len(_s.xy),))

        _s.zorder = _s.zorder

        asdf = 5

    def gen_b(_s, o1):
        """

        """

        # _s.xy = np.copy(o1.xy)
        _s.rotation = np.zeros((len(o1.xy),))
        _s.alphas = np.ones(shape=(_s.gi['frames_tot']))

        peaks_inds = scipy.signal.find_peaks(o1.xy_t[:, 1], height=3, distance=10)[0]  # OBS height needs tuning!!!  31
        '''Above line could be moved to static. Let it compute all peaks'''
        peaks_inds -= 5  # This makes them appear sooner
        neg_inds = np.where(peaks_inds < 0)[0]
        if len(neg_inds) > 0:
            peaks_inds[neg_inds] = 0

        _s.xy_t, _s.alphas, _s.rotation = foam_b(o1, peaks_inds)
        _s.zorder += 5   # Potentially this will need to be changed dynamically

        _s.xy = np.copy(_s.xy_t)
        _s.xy *= _s.o0.gi.distance_mult[_s.z_key]
        _s.xy[:, 0] += _s.gi['ld'][0] + _s.gi['o1_left_start_z']  # last one should be removed ev
        _s.xy[:, 1] += _s.gi['ld'][1]  # - xy[0, 1]
        _s.xy[:, 1] += 20

        _s.scale = np.ones(shape=(len(_s.xy),))

        # _s.xy[:, 1] += 1500  # WTF

    def gen_f(_s, o1s):
        """

        """

        '''indicies where y-tangent is at max'''
        # if int(o1.id_s[0]) + 5 >= (P.NUM_X - 1):

        _s.xy_t, _s.alphas, _s.rotation, _s.scale = foam_f(_s, o1s)  # NEED TO SHRINK GERSTNER WAVE WHEN IT BREAKS
        _s.zorder += 2530  # 1000 Z = 10 P.NUM_Z

        _s.xy = np.copy(_s.xy_t)
        _s.xy *= _s.o0.gi.distance_mult[_s.z_key]
        _s.xy[:, 0] += _s.gi['ld'][0] + _s.gi['o1_left_start_z']  # last one should be removed ev
        _s.xy[:, 1] += _s.gi['ld'][1]  # - xy[0, 1]
        # _s.xy[:, 1] += 10

        _s.scale = np.copy(o1s.scale)
        _s.scale = min_max_normalization(_s.scale, y_range=[0.6, 1])
        # _s.scale = np.ones((len(o1.scale),))
        # o1.scale = np.ones((len(o1.scale),))

    def gen_r(_s, o1):

        asdf = 5





