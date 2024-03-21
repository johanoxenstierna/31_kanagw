
import numpy as np
import random
from copy import deepcopy
from src.gen_extent_triangles import *
# from projectiles.src.gen_trig_fun import gen_scale_lds
import matplotlib.transforms as mtransforms
import P as P


class AbstractObject:
    """
    This class is supposed to be ridicilously simple
    Obs doesn't have access to the ax itself, only the info about it.
    Obs if an object is to change its movement it needs a new layer instance.
    """

    def __init__(_s):
        _s.drawn = 0  # 0: not drawn, 1: start drawing, 2. continue drawing, 3. end drawing, 4: dynamic flag usage
        _s.clock = 0
        _s.frame_ss = None
        _s.index_axs0 = None
        _s.pic = None

    def set_clock(_s, i):
        """
        The layer classes don't have access to the ax, so
        this essentially tells the ax what to do.
        """

        if i == _s.frame_ss[0]:
            _s.drawn = 1
        elif i > _s.frame_ss[0] and i < _s.frame_ss[1]:
            _s.drawn = 2  # continue. needed bcs ani_update_step will create a new axs0 otherwise
            _s.clock += 1
        elif i == _s.frame_ss[1]:
            _s.drawn = 3  # end drawing
            _s.clock = 0  # ONLY PLACE WHERE RESET
        else:  # NEEDED BCS OTHERWISE _s.drawn just stays on 3
            _s.drawn = 0

    def ani_update_step(_s, ax0, axs0, axs1, object_type=None):
        """
        Based on the drawn condition, draw, remove
        If it's drawn, return True (used in animation loop)
        OBS major bug discovered: axs0.pop(index_axs0) OBVIOUSLY results in that all index_axs0 after popped get
        screwed.
        Returns the following index:
        0: don't draw
        1: draw (will result in warp_affine)
        2: ax has just been removed, so decrement all index_axs0

        TODO: _s.drawn and _s.drawBool one of them are clearly redundant.
        """

        if _s.drawn == 0:  # not drawn, for some reason this is necessary to keep
            return 0, None
        elif _s.drawn == 1: # start
            _s.index_axs0 = len(axs0)

            '''This is where picture copy is created'''
            axs0.append(ax0.imshow(_s.pic, zorder=_s.zorder, alpha=0, origin='lower',
                                   filternorm=False))  # , extent=[0, 14, 0, 19]))

            _s.ax0 = axs0[_s.index_axs0]

            return 1, None
        elif _s.drawn == 2:  # continue drawing
            '''NEW: Checks whether sp is within bounds for ars'''
            # _s.zorder = int(_s.temp)
            return 1, None
        elif _s.drawn == 3:  # end drawing. OBS ONLY axs0
            try:
                axs0[_s.index_axs0].remove()  # might save CPU-time
                axs0.pop(_s.index_axs0)  # OBS OBS!!! MAKES axs0 shorter hence all items after index_axs0 now WRONG
                _s.ax0 = None
            except:
                raise Exception("ani_update_step CANT REMOVE AX")
            index_removed = _s.index_axs0
            _s.index_axs0 = None  # THIS IS NEEDED BUT NOT SURE WHY
            return 2, index_removed


class AbstractSSS:
    """
    class for all objects that have o0 as parent
    """

    def __init__(_s, sh, id):
        # _s.occupied = False
        _s.o0 = sh
        _s.frame_ss = [None, None]
        _s.id = id
        # _s.pic = pic  # shouldnt be needed here

    def set_frame_ss(_s, ii, NUM_FRAMES):
        """
        OBS USED BY Smokes and Spl, which are children to ship. Generates frame_ss, scale_ss
        OBS UPDATE: frame_ss reduced by 1 in length to make sure index not exceeded
        """

        assert(ii + NUM_FRAMES)
        _s.gi['frame_ss'] = [ii, ii + NUM_FRAMES]    # OVERWRITES
        _s.frame_ss = _s.gi['frame_ss']  # THIS IS GLOBAL i (hence useless for e.g. ship.extent)

    # def get_ld_ss(_s, ssas):
    #     """
    #     scale_ship_at_start
    #     ONLY CALLED ONE TIME WHEN OBJECT IS TO BE INITED (HENCE SHIP.CLOCK CAN BE USED)
    #     BE CAREFUL WITH WHAT FRAME IS BEING SPECIFIED. FRAME_SS IS WRT GLOBAL i, BUT ship.extent is WRT ship.clock
    #     """
    #     # extent_ship_at_start = _s.ship.extent[_s.gi['frame_ss'][0]]  # probably wrong
    #     extent_ship_at_init = _s.ship.extent[_s.ship.clock]  # MUCH BETTER
    #     # extent_ship_at_stop = _s.ship.extent[_s.gi['frame_ss'][1]]
    #
    #
    #     # FIRST SET IT TO BE SAME LD AS SHIP AT FRAME (DOES NOT MAKE SENSE TO USE SHIP STOP HERE SINCE THESE OBJ ARE NOT MOVING WITH SHIP
    #     ld_ss = [[extent_ship_at_init[0], extent_ship_at_init[2]],
    #              [extent_ship_at_init[0], extent_ship_at_init[2]]]
    #
    #     # ADD OFFSET
    #     try:
    #         ld_ss[0][0] += _s.gi['ld_offset_ss'][0][0] * ssas  # this is ld!
    #     except:
    #         adf = 5
    #     ld_ss[0][1] += _s.gi['ld_offset_ss'][0][1] * ssas
    #     ld_ss[1][0] += _s.gi['ld_offset_ss'][1][0] * ssas
    #     ld_ss[1][1] += _s.gi['ld_offset_ss'][1][1] * ssas  # if offset is same at beginning and end == will not move
    #
    #     # ADD RAND
    #     left_rand_start = random.randint(-_s.gi['ld_offset_rand_ss'][0][0], _s.gi['ld_offset_rand_ss'][0][0]) * ssas
    #     down_rand_start = random.randint(-_s.gi['ld_offset_rand_ss'][0][1], _s.gi['ld_offset_rand_ss'][0][1]) * ssas
    #     ld_ss[0][0] += left_rand_start
    #     ld_ss[0][1] += down_rand_start
    #     ld_ss[1][0] += left_rand_start  # same rand for both (which means proportion of pic will be same
    #     ld_ss[1][1] += down_rand_start
    #
    #     if _s.__class__.__name__ == 'Smoke':  # STOPPING VALUE CHANGES
    #
    #         ld_ss[1][0] += random.randint(-_s.gi['ld_offset_rand_ss'][1][0], _s.gi['ld_offset_rand_ss'][1][0]) * ssas  # stop left
    #         ld_ss[1][1] += random.randint(-_s.gi['ld_offset_rand_ss'][1][1], _s.gi['ld_offset_rand_ss'][1][1]) * ssas  # stop down
    #         # ld_ss[1] = [None, None]
    #         aa = 6
    #
    #     return ld_ss

    # def gen_dyn_extent_alpha(_s):  # overwritten by children
    #     pass

    def check_frame_max(_s, ii, NUM_FRAMES):  # defined in specific functions

        exceeds_frame_max = False
        how_many = 0
        if ii + NUM_FRAMES >= P.FRAMES_STOP - 20:
            exceeds_frame_max = True
            how_many = P.FRAMES_STOP - ii - 20
        return exceeds_frame_max, how_many




