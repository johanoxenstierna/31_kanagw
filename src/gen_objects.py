import os
import json
import numpy as np
from src.load_pics import load_pics
from src.genesis import _genesis
import P as P
from src.objects.o0 import O0C
from src.objects.o1 import O1C
# from src.objects.o2 import O2C


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
        _s.PATH_IMAGES = './pictures/processed/'
        # _s.ch = ch

    def gen_backgr(_s, ax_b, axs0, axs1):

        """UPDATED!!!"""
        ax_b.imshow(_s.pics['backgr_d'], zorder=1, alpha=1)  # index 0
        ax_b.axis([0, P.MAP_DIMS[0], P.MAP_DIMS[1], 0])
        ax_b.axis('off')  # TURN ON FOR FINAL

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

        """O1"""
        for o0_id, o0 in O0.items():
            # if 'O1' in o0.gi.child_names:
            O1_pics = _s.pics['O0'][o0_id]['O1']  # OBS THEY ARE DUPLICATED
            # sp_id_int = 0  # since there may be multiple f

            '''OBS from o0s perspective they are a heap. This is good longterm to 
            instead use a tensor and scatter.'''
            # for pic_key, pic in O1_pics.items():  # NUM_X.  Modulo needed here, or some way of selecting random

            for i in range(P.NUM_X):

                '''
                This will be replaced later with sampling from a prepped image
                '''
                # if i % 2 == 0:
                # pic_x_key = str(i) + '_' + 'a'
                # pic = O1_pics['a']
                # else:

                pic_x_key = str(i) + '_' + 'b'
                pic = O1_pics['b']

                pic_f = O1_pics['d']  # one f for each

                for j in range(P.NUM_Z):  # ONLY Z enumerated!
                    '''Start at bottom?'''
                    # pic_enumer = pic_key.split('_')
                    # pic_enumer = int(pic_enumer[-1])

                    if pic_x_key == '18_b':
                        adf = 5
                    pic_key = pic_x_key + '_' + str(j)
                    o1 = O1C(o1_id=pic_key, pic=pic, o0=o0)  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
                    o1.gen_static()
                    o0.O1[pic_key] = o1


                    '''Each static has two f linked to it "o1" below.
                    b=foam moving backwards, f=forwards
                    Obs they need to have a lifetime linked with wave-periods. '''
                    # if pic_x_key != '18_b':
                    pic_key = str(i) + '_' + 'd_' + str(j) + '_b'
                    o1f_b = O1C(o1_id=pic_key, pic=pic_f, o0=o0)  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
                    o1f_b.gen_b(o1)
                    o0.O1[pic_key] = o1f_b

                    '''TODO: This needs to be controlled through start frames'''
                    # if pic_x_key in ['15_b', '18_b', '20_b']:  # BECAUSE 15 STARTS AT TOP. KEY IS TO DEFINE START FRAMES
                    '''1_b is messed up'''
                    # if pic_x_key != '18_b':  # BECAUSE 15 STARTS AT TOP. KEY IS TO DEFINE START FRAMES
                    #
                    pic_key = str(i) + '_' + 'd_' + str(j) + '_f'
                    o1f_f = O1C(o1_id=pic_key, pic=pic_f, o0=o0)  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
                    o1f_f.gen_f(o1)
                    o0.O1[pic_key] = o1f_f

        return O0

