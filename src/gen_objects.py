import os
import json
import numpy as np

from matplotlib.pyplot import imread
from src.load_pics import load_pics
from src.genesis import _genesis
import P as P

from src.objects.o0 import O0C
from src.objects.o1 import O1C
# from src.objects.o2 import O2C
from pictures import prep_k0


class GenObjects:

    """
    OBS this time it's only the background that is being ax.shown here. The other ax objects are added and
    deleted within the animation loop to save CPU-time.
    Each pic is tied to exactly 1 class instance and that class instance takes info from either o0 parent
    or other.
    """

    def __init__(_s):
        _s.pics = load_pics()
        _s.gis = _genesis()
        # _s.PATH_IMAGES = './pictures/processed/'
        # _s.ch = ch

    def gen_backgr(_s, ax_b, axs0, axs1):

        """UPDATED!!!"""
        ax_b.imshow(_s.pics['backgr'], zorder=1, alpha=1)  # index 0
        ax_b.axis([0, P.MAP_DIMS[0], P.MAP_DIMS[1], 0])
        ax_b.axis('off')  # TURN ON FOR FINAL
        ax_b.set_axis_off()

    def gen_O0(_s):
        """
        Base objects.
        """
        O0 = {}
        for o0_id in P.O0_TO_SHOW:  # number_id
            o0_gi = _s.gis[o0_id]
            O0[o0_id] = O0C(pic=None, gi=o0_gi)  # No pic CURRENTLY

        return O0

    def gen_O1_new(_s, O0):
        """
        This function may eventually be run to generate cut up images beforehand, if it takes too long.
        TODO: Need to think about padding the outsides
        Also need to think about using top-down or sheared k0
        Also need to think about whether O1s should be combined based on wave directions.
        Probably not, and bcs of that everything has to be particle-based - everything
        that is shown needs to be explainable through the generated waves.
        """

        '''THESE ARE TEMPORARY. REMOVE WHEN PADDING SORTED'''

        k0 = imread('./pictures/k0.png')
        k0 = np.flipud(k0)  # essential

        d = 0
        if P.COMPLEXITY == 0:  # needed cuz overlap between dots may be of interest
            d = int(1000 / P.NUM_X)
        elif P.COMPLEXITY == 1:
            d = int(1500 / P.NUM_X)  # OBS check against alpha.

        if d % 2 != 0:  # this problem is likely due to there not being any picture to sample from.
            d += 1

        # These are now from bottom. WHERE THEY APPEAR ON MAP
        BOT_Z = int(d/2) + 1  # If start_z = 25, that means diameter max is 49
        TOP_Z = 200  # this plus half diameter

        # pend del
        if BOT_Z < int(d / 2):
            print("inds_z[0]: " + str(BOT_Z), "   d: " + str(d))
            raise Exception("d too large")

        '''indexing has to be identical for prepping k0 cuts and generaing the o1 objects'''
        inds_x = np.linspace(start=100, stop=1150, num=P.NUM_X, dtype=int)
        inds_z = np.linspace(start=BOT_Z, stop=TOP_Z, num=P.NUM_Z, dtype=int)

        prep_k0.cut_k0(k0, inds_x, inds_z, d)
        c_, d_ = prep_k0.get_c_d(k0, d)
        print("DIAMETER: " + str(d))

        for i in range(len(inds_x)):
            for j in range(len(inds_z)):  # smallest ind = bottom
                ind_x = inds_x[i]
                ind_z = inds_z[j]

                type = 'static'
                file_name = str(ind_x) + '_' + str(ind_z) + '.npy'
                id_static = str(i) + '_' + str(j) + '_' + type
                if P.COMPLEXITY == 0:
                    # pic_static = _s.pics['O0']['waves']['O1']['d']
                    pic_static = imread('./pictures/waves/O1/d.png')
                elif P.COMPLEXITY == 1:
                    pic_static = np.load('./pictures/k0_cut/' + file_name)

                '''OBS this combines multiple G waves'''
                o1 = O1C(o1_id=id_static, pic=pic_static, o0=O0['waves'], type=type)  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
                o1.gen_static()
                O0['waves'].O1[id_static] = o1

                '''Each static has two f linked to it "o1" below.
                b=foam moving backwards, f=forwards
                Obs they need to have a lifetime linked with wave-periods. '''

                type = 'f_b'  # THESE GUYS SHOULD ONLY START AFTER BREAK. BEFORE IS WRONG
                id_f_b = str(i) + '_' + str(j) + '_' + type
                o1f_b = O1C(o1_id=id_f_b, pic=c_, o0=O0['waves'], type=type)  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
                o1f_b.gen_b(o1)
                O0['waves'].O1[id_f_b] = o1f_b

                # type = 'f_f'  # NOT USED FOR SMALL ONES
                # id_f_f = str(i) + '_' + str(j) + '_' + type
                # o1f_f = O1C(o1_id=id_f_f, pic=d_, o0=O0['waves'], type=type)  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
                # o1f_f.gen_f(o1)
                # O0['waves'].O1[id_f_f] = o1f_f

                adf = 5

            print(i)

        asdf =5

        return O0

    # def gen_O1(_s, O0):
    #
    #     """O1"""
    #     for o0_id, o0 in O0.items():
    #         # if 'O1' in o0.gi.child_names:
    #         O1_pics = _s.pics['O0'][o0_id]['O1']  # OBS THEY ARE DUPLICATED
    #         # sp_id_int = 0  # since there may be multiple f
    #
    #         '''OBS from o0s perspective they are a heap. This is good longterm to
    #         instead use a tensor and scatter.'''
    #         # for pic_key, pic in O1_pics.items():  # NUM_X.  Modulo needed here, or some way of selecting random
    #
    #         for x in range(P.NUM_X):
    #
    #             '''
    #             This will be replaced later with sampling from a prepped image
    #             '''
    #             # if i % 2 == 0:
    #             # pic_x_key = str(i) + '_' + 'a'
    #             # pic = O1_pics['a']
    #             # else:
    #             # pic_key = 'b'
    #             x_key = str(x)
    #
    #
    #             # pic_f = O1_pics['d']  # one f for each. FOR NOW
    #
    #             for z in range(P.NUM_Z):  # ONLY Z enumerated!
    #                 '''Start at bottom?'''
    #                 # pic_enumer = pic_key.split('_')
    #                 # pic_enumer = int(pic_enumer[-1])
    #
    #                 z_key = str(z)
    #
    #                 pic_key = 'b'
    #                 type = 'static'
    #                 id_static = x_key + '_' + z_key + '_' + type
    #                 pic_static = O1_pics[pic_key]
    #                 o1 = O1C(o1_id=id_static, pic=pic_static, o0=o0, type=type)  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
    #                 o1.gen_static()
    #                 o0.O1[id_static] = o1
    #
    #
    #                 '''Each static has two f linked to it "o1" below.
    #                 b=foam moving backwards, f=forwards
    #                 Obs they need to have a lifetime linked with wave-periods. '''
    #
    #                 pic_key = 'c'
    #                 type = 'f_b'
    #                 id_f_b = x_key + '_' + z_key + '_' + type
    #                 pic_f_b = O1_pics[pic_key]
    #                 o1f_b = O1C(o1_id=id_f_b, pic=pic_f_b, o0=o0, type=type)  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
    #                 o1f_b.gen_b(o1)
    #                 o0.O1[id_f_b] = o1f_b
    #
    #                 # pic_key = 'd'
    #                 # type = 'f_f'
    #                 # id_f_f = x_key + '_' + z_key + '_' + type
    #                 # pic_f_f = O1_pics[pic_key]
    #                 # o1f_f = O1C(o1_id=id_f_f, pic=pic_f_f, o0=o0, type=type)  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
    #                 # o1f_f.gen_f(o1)
    #                 # o0.O1[id_f_f] = o1f_f
    #
    #     return O0
