
import numpy as np
import P
import scipy
from scipy.stats import beta
from src.trig_functions import min_max_normalization


def gen_stns():
    """
    Cant shift it cuz new values need to be generated each z
    """

    PATH_OUT = './O0s_info/stns_ZX.npy'


    # stns_z = beta.pdf(x=np.linspace(0, 1, P.NUM_Z), a=5, b=5, loc=0)  # a>b BIGGEST FURTHEST AWAY
    # stns_z = min_max_normalization(stns_z, y_range=[3, 4])  # OBS BIGGEST IND IS FURTEST FROM SCREEN
    # peak = scipy.signal.find_peaks(stns_z)[0][-1]
    # # stns_z0[peak:] *= np.exp(np.linspace(start=0, stop=-2, num=P.NUM_Z - peak))

    '''Break distribution needs to go inside z bounds. So var'''
    # start_z = 0
    # stop_z = P.NUM_Z
    # num_z_stn = P.NUM_Z
    #
    start = 0
    stop = P.NUM_X
    # num_stn = P.NUM_X


    # if P.NUM_Z > 7:
    #     start_z = int(0.3 * P.NUM_Z)
    #     stop_z = int(0.7 * P.NUM_Z)
    #     num_stn = stop_z - start_z

    # if P.NUM_X > 7:
    #     start_x = int(0.3 * P.NUM_X)
    #     stop_x = int(0.7 * P.NUM_X)
    #     num_stn = stop_x - start_x

    # stn_sub = range(num_stn)

    '''Need to work with EITHER x or z'''

    # A = np.linspace(15, 10, num=P.NUM_X)  # Z decides how thick break is
    # B = np.linspace(10, 15, num=P.NUM_X)

    LOW_BOUND_y = 1

    stns_ZX = np.full(shape=(P.FRAMES_TOT, P.NUM_Z, P.NUM_X), fill_value=LOW_BOUND_y, dtype=np.float16)

    '''STN X: One stn array per z: wave breaks in middle of x axis '''
    num_stn = P.NUM_Z

    A = np.linspace(15, 10, num=P.NUM_Z)  # X decides how thick break is
    B = np.linspace(10, 15, num=P.NUM_Z)

    for i in range(num_stn):  # OBS 0 is closest to screen!

        stns_x = beta.pdf(x=np.linspace(0, 1, P.NUM_X), a=A[i], b=B[i], loc=0)
        stns_x = min_max_normalization(stns_x, y_range=[LOW_BOUND_y, 3])  # MAINLY TO PREVENT LEFT FROM BREAKING
        peak = scipy.signal.find_peaks(stns_x)[0][-1]
        stns_x[peak:] *= np.exp(np.linspace(start=0, stop=-1.5, num=P.NUM_X - peak))

        stns_ZX[0, i, :] += 0.85 * stns_x

    '''STN Z: One stn array per x: wave breaks in middle of x axis '''
    num_stn = P.NUM_X
    A = np.linspace(15, 10, num=P.NUM_X)  # X decides how thick break is
    B = np.linspace(10, 15, num=P.NUM_X)

    for i in range(num_stn):  # OBS 0 is closest to screen!

        stns_z = beta.pdf(x=np.linspace(0, 1, P.NUM_Z), a=A[i], b=B[i], loc=0)
        stns_z = min_max_normalization(stns_z, y_range=[LOW_BOUND_y, 3])  # MAINLY TO PREVENT LEFT FROM BREAKING
        # stns_x0 = min_max_normalization(w0 + w1 + w2, y_range=[0.5, 1.8])
        peak = scipy.signal.find_peaks(stns_z)[0][-1]
        # aa = np.exp(np.linspace(start=0, stop=-2, num=P.NUM_X - peak))
        # stns_z[peak:] *= np.exp(np.linspace(start=0, stop=-1.5, num=P.NUM_Z - peak))
        stns_z[:peak] *= np.exp(np.linspace(start=-1.5, stop=0, num=peak))

        stns_ZX[0, :, i] += 0.15 * stns_z

    # for i in range(P.FRAMES_TOT):
    # saa = np.flipud(stns_ZX[0, :, :])
    # A55 = np.rot90(stns_ZX[0, :, :])
    # ggg = np.fliplr(A55)
    aa = 5
    # for i in range(P.NUM_Z):
    #     for j in range(P.NUM_X):
    #         stn_z = stns_z[i]
    #         stn_x = stns_x[j]
    #         stn_zx = 0.5 * stn_z + 0.5 * stn_x
    #         stns_zx[i, j] = stn_zx  # OBS BIGGEST ROW IS FURTEST FROM SCREEN  (i=0 => BOTTOM)

    np.save(PATH_OUT, stns_ZX)

    return stns_ZX




if __name__ == "__main__":
    stns_ZX = gen_stns()


# def orig():
#     stns_zx0 = np.zeros(shape=(P.NUM_Z, P.NUM_X))
#     stns_zx1 = np.zeros(shape=(P.NUM_Z, P.NUM_X))
#
#     stns_z0 = beta.pdf(x=np.linspace(0, 1, P.NUM_Z), a=5, b=5, loc=0)  # a>b BIGGEST FURTHEST AWAY
#     stns_z0 = min_max_normalization(stns_z0, y_range=[3, 4])  # OBS BIGGEST IND IS FURTEST FROM SCREEN
#     peak = scipy.signal.find_peaks(stns_z0)[0][-1]
#     # stns_z0[peak:] *= np.exp(np.linspace(start=0, stop=-2, num=P.NUM_Z - peak))
#
#     stns_x0 = beta.pdf(x=np.linspace(0, 1, P.NUM_X), a=10, b=10, loc=0)
#     stns_x0 = min_max_normalization(stns_x0, y_range=[0.1, 3])  # MAINLY TO PREVENT LEFT FROM BREAKING
#     # stns_x0 = min_max_normalization(w0 + w1 + w2, y_range=[0.5, 1.8])
#     peak = scipy.signal.find_peaks(stns_x0)[0][-1]
#     stns_x0[peak:] *= np.exp(np.linspace(start=0, stop=-10, num=P.NUM_X - peak))
#
#     stns_z1 = beta.pdf(x=np.linspace(0, 1, P.NUM_Z), a=5, b=2, loc=0)  # a>b BIGGEST FURTHEST AWAY
#     stns_z1 = min_max_normalization(stns_z1, y_range=[0.2, 1.5])  # OBS BIGGEST IND IS FURTEST FROM SCREEN
#     peak = scipy.signal.find_peaks(stns_z1)[0][0]
#     stns_z1[peak:] *= np.exp(np.linspace(start=0, stop=-5, num=P.NUM_Z - peak))
#
#     stns_x1 = beta.pdf(x=np.linspace(0, 1, P.NUM_X), a=4, b=5, loc=0)
#     stns_x1 = min_max_normalization(stns_x1, y_range=[0.2, 1.5])
#     peak = scipy.signal.find_peaks(stns_x1)[0][0]
#     stns_x1[peak:] = np.exp(np.linspace(start=0, stop=-5, num=P.NUM_X - peak))
#
#     for i in range(P.NUM_Z):
#         for j in range(P.NUM_X):
#             stn_z = stns_z0[i]
#             stn_x = stns_x0[j]
#             stn_zx = 0.5 * stn_z + 0.5 * stn_x
#             stns_zx0[i, j] = stn_zx  # OBS BIGGEST ROW IS FURTEST FROM SCREEN  (i=0 => BOTTOM)
#
#             stn_z = stns_z1[i]
#             stn_x = stns_x1[j]
#             stn_zx = 0.2 * stn_z + 0.8 * stn_x
#             stns_zx1[i, j] = stn_zx  # OBS BIGGEST ROW IS FURTEST FROM SCREEN (i=0 => BOTTOM)
#
#     return stns_zx0