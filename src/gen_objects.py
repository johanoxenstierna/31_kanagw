import os
import json
import numpy as np
import random
import time
import psutil

from matplotlib.pyplot import imread
from src.load_pics import load_pics
from src.genesis import _genesis
import P as P

from src.objects.o0 import O0C
from src.objects.o1 import O1C
# from src.objects.o2 import O2C
from src.trig_functions import min_max_normalization
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

    def gen_O1(_s, O0):
        """
        This function may eventually be run to generate cut up images beforehand, if it takes too long.
        TODO: Need to think about padding the outsides
        Also need to think about using top-down or sheared k0
        Also need to think about whether O1s should be combined based on wave directions.
        Probably not, and bcs of that everything has to be particle-based - everything
        that is shown needs to be explainable through the generated waves.
        """

        '''THESE ARE TEMPORARY (haha no). REMOVE WHEN PADDING SORTED'''

        k0 = imread('./pictures/k0.png')
        k0 = np.flipud(k0)  # essential

        d = 0  # diameter
        if P.COMPLEXITY == 0:  # needed cuz overlap between dots may be of interest
            d = int(1000 / P.NUM_X)
        elif P.COMPLEXITY == 1:
            d = int(1500 / P.NUM_X)  # OBS check against alpha.
            d = max(20, d)

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
        pxls_x = np.linspace(start=100, stop=1150, num=P.NUM_X, dtype=int)
        pxls_z = np.linspace(start=BOT_Z, stop=TOP_Z, num=P.NUM_Z, dtype=int)

        prep_k0.cut_k0(k0, pxls_x, pxls_z, d)
        b_, f_ = prep_k0.get_c_d(k0, d)

        '''KANAGAWA FRACTALS'''
        if P.A_K:
            R_, R_inds_used = prep_k0.get_kanagawa_fractals()

        print("DIAMETER: " + str(d))

        '''
        Update: cant run static and f in same loop anymore cuz landing needed
        CPU time DOUBLED
        '''
        time0 = time.time()
        for i in range(P.NUM_X):  # time scales linearly with num particles
            print(str(i))
            # print(str(i) + '  ' + str(psutil.swap_memory()))
            for j in range(P.NUM_Z):  # smallest ind = bottom
                pxl_x = pxls_x[i]
                pxl_z = pxls_z[j]

                type = 'static'
                file_name = str(pxl_x) + '_' + str(pxl_z) + '.npy'
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

                # type = 'f_b'  # THESE GUYS SHOULD ONLY START AFTER BREAK. BEFORE IS WRONG
                # id_f_b = str(i) + '_' + str(j) + '_' + type
                # o1f_b = O1C(o1_id=id_f_b, pic=b_, o0=O0['waves'], type=type)  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
                # o1f_b.gen_b(o1)
                # O0['waves'].O1[id_f_b] = o1f_b

        time_static = time.time() - time0
        print("time static: " + str(time_static))

        time0 = time.time()
        for i in range(P.NUM_X):
            # print(i)
            for j in range(P.NUM_Z):  # smallest ind = bottom
                if P.A_F:
                    o1s = O0['waves'].O1[str(i) + '_' + str(j) + '_static']
                    type = 'f'  # NOT USED FOR SMALL ONES
                    id_f = str(i) + '_' + str(j) + '_' + type
                    o1f = O1C(o1_id=id_f, pic=f_, o0=O0['waves'], type=type)  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?

                    # time0 = time.time()

                    if i + 5 >= (P.NUM_X - 1):
                        pass
                    else:
                        o1f.o1_sib = O0['waves'].O1[str(i + 5) + '_' + str(j) + '_static']

                    o1f.gen_f(o1s)
                    # print(time.time() - time0)

                    O0['waves'].O1[id_f] = o1f

                    # if P.A_K:
                    #     if _s.gis['waves'].TH[0, j, i] == 2 and random.random() < 0.5:
                    #
                    #     # if (j, i) in R_inds_used:  # totally ok to have inds too large: they just wont appear in smaller animation
                    #         type = 'r'  # NOT USED FOR SMALL ONES
                    #         id_r = str(i) + '_' + str(j) + '_' + type
                    #         # r_ = R_[str(j) + '_' + str(i)]
                    #         _, r_ = random.choice(list(R_.items()))
                    #         o1r = O1C(o1_id=id_r, pic=r_, o0=O0['waves'], type=type)
                    #         o1r.gen_f(o1f)
                    #         # o1r.gen_r(o1)
                    #         o1r.scale = min_max_normalization(o1r.scale, y_range=[0.5, 1.3])
                    #         # o1r.alphas = min_max_normalization(o1r.alphas, y_range=[0, 1])  # this one needs to be changed
                    #         o1r.rotation *= 1.4
                    #         # o1r.gen_r(o1f)
                    #         O0['waves'].O1[id_r] = o1r
                    #         # o1r.zorder += 2000

                adf = 5

            print(i)

        time_f = time.time() - time0

        print("time static: " + str(time_static))
        print("time f: " + str(time_f))
        asdf =5

        return O0

    def connect_siblings(_s, O0):

        for o1_key, o1 in O0['waves'].O1.items(): #  this loops over the f.
            if o1.type == 'static':
                continue

            if int(o1.id_s[0]) + 5 >= (P.NUM_X - 1):
                continue

            id_sib = str(int(o1.id_s[0]) + 5) + '_' + str(int(o1.id_s[1]) + 0) + '_static'

            o1_sib = O0['waves'].O1[id_sib]

            o1.o1_sib = o1_sib


