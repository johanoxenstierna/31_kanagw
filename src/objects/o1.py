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
        _s.z_key = int(_s.id_s[1])

        _s.zorder = 100 + _s.z_key + _s.x_key  # needs x key as well

        _s.o0 = o0  # parent
        _s.pic = pic  # the png
        _s.type = type
        _s.gi = deepcopy(o0.gi.o1_gi)  # OBS!  COPY SHOULD NOT BE THERE. SHOULD BE READ-ONLY.OK WHILE FEW OBJECTS.
        # ONLY OBJECTS THAT ARE MUTABLE ARE TO BE COPIED

        AbstractSSS.__init__(_s, o0, o1_id)

        _s.O2 = {}
        _s.alphas = None

        '''
        
        '''

        _s.gi['init_frames'] = [_s.o0.gi.o1_init_frames[0]]  # same for all

        _s.gi['ld'][0] = _s.o0.gi.o1_left_x[_s.x_key] + _s.o0.gi.o1_left_z[_s.z_key]  # last one is shear! probably removed later
        _s.gi['ld'][1] = _s.o0.gi.o1_down_z[_s.z_key]

        if P.COMPLEXITY == 0:
            pass
        elif P.COMPLEXITY == 1:
            _s.gi['ld'][0] += random.randint(-15, 15)
            _s.gi['ld'][1] += random.randint(-10, 10)
        # _s.gi['steepness'] = _s.o0.gi.o1_steepnessess_z[_s.z_id] #+ np.random.randint(low=0, high=50, size=1)[0]
        # _s.gi['steepness'] = _s.o0.gi.stns_zx[_s.z_key, _s.x_key] #+ np.random.randint(low=0, high=50, size=1)[0]
        _s.gi['o1_left_start_z'] = _s.o0.gi.o1_left_starts_z[_s.z_key] #+ np.random.randint(low=0, high=50, size=1)[0]

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
        _s.alphas, _s.rotation, _s.peaks = gerstner_waves(o1=_s, o0=_s.o0)
        # _s.alphas = np.zeros(shape=(_s.gi['frames_tot']))
        # _s.xy[:, 1] *= -1  # flip it.

        '''shifting '''
        _s.xy = np.copy(_s.xy_t)
        _s.xy *= _s.o0.gi.distance_mult[_s.z_key]
        _s.xy[:, 0] += _s.gi['ld'][0] + _s.gi['o1_left_start_z']  # last one should be removed ev
        _s.xy[:, 1] += _s.gi['ld'][1]  # - xy[0, 1]

        # _s.xy[:, 1] += 50

        # _s.o0.populate_T(_s.xy_t, _s.xy, _s.dxy)

        _s.zorder = _s.zorder #

        asdf = 5

    def gen_b(_s, o1):
        """

        """

        # _s.xy = np.copy(o1.xy)
        _s.rotation = np.zeros((len(o1.xy),))
        _s.alphas = np.ones(shape=(_s.gi['frames_tot']))

        peaks_inds = scipy.signal.find_peaks(o1.xy_t[:, 1], height=15, distance=50)[0]
        peaks_inds -= 5
        neg_inds = np.where(peaks_inds < 0)[0]
        if len(neg_inds) > 0:
            peaks_inds[neg_inds] = 0

        _s.xy_t, _s.alphas, _s.rotation = foam_b(o1, peaks_inds)
        _s.zorder += 5   # Potentially this will need to be changed dynamically

        _s.xy = np.copy(_s.xy_t)
        _s.xy *= _s.o0.gi.distance_mult[_s.z_key]
        _s.xy[:, 0] += _s.gi['ld'][0] + _s.gi['o1_left_start_z']  # last one should be removed ev
        _s.xy[:, 1] += _s.gi['ld'][1]  # - xy[0, 1]
        _s.xy[:, 1] += 5

    def gen_f(_s, o1):
        """
        self is o1f_f and o1 is the base
        """

        '''indicies where y-tangent is at max'''
        # aa = np.where(o1.dxy[:, 1] > )

        peaks_inds = scipy.signal.find_peaks(o1.xy_t[:, 1], height=20, distance=50)[0]
        peaks_inds -= 5
        neg_inds = np.where(peaks_inds < 0)[0]
        if len(neg_inds) > 0:
            peaks_inds[neg_inds] = 0

        # if len(peaks_inds) < 1:
        #     _s.gi['init_frames'] = [3]
        #     _s.gi['frames_tot'] = 2
        #     _s.xy = np.array([[4, 4], [5, 5]])
        #     _s.alphas = np.array([0.5, 0.5])
        #     _s.zorder = 1
        #     _s.rotation = np.array([0, 0])
        # else:
            # _s.gi['init_frames'] = [peaks_inds[0]]
            # _s.gi['frames_tot'] = 50

        _s.xy_t, _s.alphas, _s.rotation = foam_f(o1, peaks_inds)  # NEED TO SHRINK GERSTNER WAVE WHEN IT BREAKS
        _s.zorder += 20

        _s.xy = np.copy(_s.xy_t)
        _s.xy *= _s.o0.gi.distance_mult[_s.z_key]
        _s.xy[:, 0] += _s.gi['ld'][0] + _s.gi['o1_left_start_z']  # last one should be removed ev
        _s.xy[:, 1] += _s.gi['ld'][1]  # - xy[0, 1]
        # _s.xy[:, 1] += 10

    # def set_frame_stop_to_sp_max(_s):
    #     """Loop through sps and set max to frame_stop"""
    #
    #     _max = 0
    #     for sp_id, sp in _s.sps.items():
    #         if sp.frame_ss[1] > _max:
    #             _max = sp.frame_ss[1]
    #
    #     _s.frame_ss[1] = deepcopy(_max) + 5



    # def finish_info(_s):
    #     """Separated from _init_ bcs extra things may need to be finished in viewer."""
    #
    #     _s.alphas = np.ones(shape=(_s.gi['frames_tot']))
    #
    #     # if _s.o0.id == 'projectiles':
    #     #
    #     #     id_int = int(_s.id[-1])  # OBS. Used by o1_down_offsets
    #     #     # _s.gi['ld'][1] += _s.o0.gi.o1_down_offsets[id_int]  # NOT GOOD. dont change parameters
    #     #
    #     #     '''30_ xys and thetas based on direction'''
    #     #     XY = np.zeros(shape=(_s.gi['frames_tot'], 2))
    #     #     # XY[:, 1] = _s.gi['ld'][1]  # y never changes
    #     #
    #     #     rand_0 = np.random.choice([-1, 1])
    #     #     rand_1 = np.random.randint(low=1, high=6, size=1)[0]
    #     #
    #     #     X = rand_0 * np.sin(np.linspace(0, rand_1 * np.pi, num=len(XY)))
    #     #     rot = rand_0 * 0.5 * np.cos(np.linspace(0, rand_1 * np.pi, num=len(XY)))
    #     #     XY[:, 0] = _s.gi['ld'][0] + X * np.random.randint(low=5, high=30, size=1)[0]
    #     #     XY[:, 1] = _s.gi['ld'][1] + _s.o0.gi.o1_down_offsets[id_int] - np.sin(np.linspace(0, 0.5 * np.pi, num=len(XY))) * 700
    #     #     # if np.min(XY[:, 1]) < 0:
    #     #     #     raise Exception("o2 going out of frame")
    #     #     _s.XY = XY
    #     #
    #     #     _s.rot = rot
    #     #     _s.cmap = random.choice(['afmhot', 'Wistia', 'cool', 'hsv', 'summer'])
    #     #     # _s.cmap = random.choice(['hsv'])
    #     #
    #     #     _s.alphas = gen_alpha(_s, _type='o1_projectiles')
    #     #
    #     # elif _s.o0.id == 'clouds':
    #     #
    #     #     # id_int = int(_s.id[-1])  # OBS. Used by o1_down_offsets
    #     #     # _s.gi['ld'][1] += _s.o0.gi.o1_down_offsets[id_int]  # NOT GOOD. dont change parameters
    #     #
    #     #     '''30_ xys and thetas based on direction'''
    #     #     XY = np.zeros(shape=(_s.gi['frames_tot'], 2))
    #     #     # XY[:, 1] = _s.gi['ld'][1]  # y never changes
    #     #
    #     #     X = np.linspace(0, 300, num=len(XY))
    #     #     Y = np.linspace(0, 100, num=len(XY))
    #     #
    #     #     left_offset = np.random.randint(low=0, high=100, size=1)[0]
    #     #     down_offset = np.random.randint(low=0, high=50, size=1)[0]
    #     #     XY[:, 0] = _s.gi['ld'][0] + X + left_offset
    #     #     XY[:, 1] = _s.gi['ld'][1] - Y + down_offset
    #     #     _s.XY = XY
    #     #
    #     #     rot = np.linspace(0, 0.2 * np.pi, num=len(XY))
    #     #     _s.rot = rot
    #     #
    #     #     _s.scale = np.linspace(0.5, 1, num=len(XY))
    #     #
    #     #     # _s.alphas = np.full(shape=(len(XY),), fill_value=1)
    #     #     _s.alphas = gen_alpha(_s, _type='o1_clouds')