from src.gen_extent_triangles import *
from src.objects.abstract import AbstractObject
import P as P
# from projectiles.src.gen_colors import gen_colors
# import copy
import numpy as np

import random


class O0C(AbstractObject):

    def __init__(_s, pic, gi):
        super().__init__()
        _s.id = gi.id
        _s.gi = gi  # IMPORTANT replaces _s.gi = ship_info
        _s.pic = pic  # NOT SCALED
        _s.O1 = {}
        _s.O1b = {}
        _s.O1f = {}

        _s.T = np.zeros(shape=(72, 128, gi.o1_gi['frames_tot']))

    def populate_T(_s, xy_t, xy, dxy):
        """
        3D Heatmap with x cols, z rows and y values.
        No shearing preferably.
        Todo: Map boundary issues
        1: Inference over several o1 will be needed.
        2: o1 tied to cell, o1f not tied to (launch) cell.
        2: Type of o1f depends on aggregate height and dxy in (launch) cell.
        3: Each o1f tied to o1. While motion is decided
        """

        # xy_p = np.zeros(shape=xy.shape, dtype=xy.dtype)
        inds_to_fill = np.where((xy[:, 0] > 0) & (xy[:, 1] > 0) & (xy[:, 0] < 1280) & (xy[:, 1] < 720))[0]  # all null inds

        t_locs = xy / 10
        t_locs[:, 1] = 72 - t_locs[:, 1]  # need to flip upside down to get
        t_locs = t_locs.astype(int)

        aa = 5