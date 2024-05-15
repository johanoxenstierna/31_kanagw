
import numpy as np
import P
import scipy
from scipy.stats import beta, multivariate_normal
from src.trig_functions import min_max_normalization, min_max_normalize_array


def gen_stns():
    """New: Use mvn"""
    PATH_OUT = './O0s_info/stns_ZX.npy'

    '''cov: more second: more x spread. '''
    rv = multivariate_normal(mean=[P.NUM_Z / 2, P.NUM_X / 2], cov=[[P.NUM_Z / 4, P.NUM_Z],
                                                                   [P.NUM_Z, P.NUM_Z * 8]])  # more=more visible
    BOUND_LO_y = 2
    BOUND_UP_y = 4
    BOUND_MI_y = 3

    x, y = np.mgrid[0:P.NUM_Z:1, 0:P.NUM_X:1]
    pos = np.dstack((x, y))
    stns_ZX = rv.pdf(pos)
    stns_ZX = stns_ZX / np.max(stns_ZX)
    H_Z = np.zeros(shape=(P.NUM_Z, P.NUM_X), dtype=np.float16)
    H_X = np.zeros(shape=(P.NUM_Z, P.NUM_X), dtype=np.float16)

    '''Normalize'''
    stns_ZX = min_max_normalize_array(stns_ZX, y_range=[BOUND_LO_y, BOUND_UP_y])

    '''STN Z: One stn array per x'''
    # stns_ZX[int(P.NUM_Z / 2), :] += 0.0001  # to make sure there is a peak
    for i in range(P.NUM_X):  # OBS 0 is closest to screen!
        stns_ZX[:, i] += 0.0001  # to make sure there is a peak
        peak = np.argmax(stns_ZX[:, i])
        stns_ZX[:peak, i] *= np.exp(np.linspace(start=-0.5, stop=0, num=peak))

        h_z = np.copy(stns_ZX[:, i])
        h_z[:peak] = 0
        H_Z[:, i] = h_z

    '''STN X: One stn array per z'''
    # stns_ZX[:, int(P.NUM_X / 2)] += 0.0001  # to make sure there is a peak
    for i in range(P.NUM_Z):  # OBS 0 is closest to screen!
        stns_ZX[i, :] += 0.0001  # to make sure there is a peak
        peak = np.argmax(stns_ZX[i, :])
        stns_ZX[i, peak:] *= np.exp(np.linspace(start=0, stop=-1.5, num=P.NUM_X - peak))

        h_x = np.copy(stns_ZX[i, :])
        h_x[peak:] = 0
        H_X[i, :] = h_x

    np.save(PATH_OUT, stns_ZX)

    SPLIT_ZX = [0.8, 0.2]

    H = np.zeros((P.NUM_Z, P.NUM_X), dtype=np.uint16)  # fall height for f ONLY f!

    inds_buildup = np.where((BOUND_LO_y <= H_Z[:, :]) & (H_Z[:, :] <= BOUND_MI_y))
    inds_break = np.where(BOUND_MI_y < H_Z[:, :])
    inds_post = np.where(H_Z[:, :] < BOUND_LO_y)

    H[inds_buildup] += int(100 * SPLIT_ZX[0])
    H[inds_break] += int(1000 * SPLIT_ZX[0])
    H[inds_post] += 0

    inds_buildup = np.where((BOUND_LO_y <= H_X[:, :]) & (H_X[:, :] <= BOUND_MI_y))
    inds_break = np.where(BOUND_MI_y < H_X[:, :])
    inds_post = np.where(H_X[:, :] < BOUND_LO_y)

    H[inds_buildup] += int(100 * SPLIT_ZX[1])
    H[inds_break] += int(1000 * SPLIT_ZX[1])
    H[inds_post] += 0

    H[np.where((H > 4) & (H < 500))] = 1
    H[np.where(H >= 500)] = 2

    stns_TZX = np.zeros(shape=(P.FRAMES_TOT, P.NUM_Z, P.NUM_X), dtype=np.float16)
    stns_TZX[0, :, :] = stns_ZX

    return stns_TZX, H


def gen_stns_old():
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

    BOUND_LO_y = 2
    BOUND_UP_y = 4
    BOUND_MI_y = 3

    # stns_ZX = np.full(shape=(P.FRAMES_TOT, P.NUM_Z, P.NUM_X), fill_value=LOW_BOUND_y, dtype=np.float16)
    stns_ZX = np.zeros(shape=(P.FRAMES_TOT, P.NUM_Z, P.NUM_X), dtype=np.float16)
    stns_Z = np.zeros(shape=(P.NUM_Z, P.NUM_X), dtype=np.float16)
    stns_X = np.zeros(shape=(P.NUM_Z, P.NUM_X), dtype=np.float16)
    H_Z = np.zeros(shape=(P.NUM_Z, P.NUM_X), dtype=np.float16)
    H_X = np.zeros(shape=(P.NUM_Z, P.NUM_X), dtype=np.float16)

    SPLIT_ZX = [0.6, 0.4]  # NOT USED BY H

    '''STN Z: One stn array per x: wave breaks in middle of x axis '''
    num_stn = P.NUM_X
    A = np.linspace(2, 5, num=P.NUM_X)  # X decides how thick break is
    B = np.linspace(5, 3, num=P.NUM_X)

    for i in range(num_stn):  # OBS 0 is closest to screen!

        stns_z = beta.pdf(x=np.linspace(0, 1, P.NUM_Z), a=A[i], b=B[i], loc=0)
        stns_z = min_max_normalization(stns_z, y_range=[BOUND_LO_y, BOUND_UP_y])  # MAINLY TO PREVENT LEFT FROM BREAKING
        peak = scipy.signal.find_peaks(stns_z)[0][-1]
        # aa = np.exp(np.linspace(start=0, stop=-2, num=P.NUM_X - peak))
        # stns_z[peak:] *= np.exp(np.linspace(start=0, stop=-1.5, num=P.NUM_Z - peak))
        stns_z[:peak] *= np.exp(np.linspace(start=-1.5, stop=0, num=peak))
        h_z = np.copy(stns_z)
        h_z[:peak] = 0

        stns_Z[:, i] = stns_z
        H_Z[:, i] = h_z

    '''STN X: One stn array per z: wave breaks in middle of x axis '''
    num_stn = P.NUM_Z

    A = np.linspace(2, 5, num=P.NUM_Z)  # X decides how thick break is
    B = np.linspace(5, 2, num=P.NUM_Z)

    for i in range(num_stn):  # OBS 0 is closest to screen!

        stns_x = beta.pdf(x=np.linspace(0, 1, P.NUM_X), a=A[i], b=B[i], loc=0)
        stns_x = min_max_normalization(stns_x, y_range=[BOUND_LO_y, BOUND_UP_y])  # MAINLY TO PREVENT LEFT FROM BREAKING
        peak = scipy.signal.find_peaks(stns_x)[0][-1]
        stns_x[peak:] *= np.exp(np.linspace(start=0, stop=-1.5, num=P.NUM_X - peak))
        h_x = np.copy(stns_x)
        h_x[peak:] = 0

        stns_X[i, :] = stns_x
        H_X[i, :] = h_x


    stns_ZX[0, :, :] = SPLIT_ZX[0] * stns_Z + SPLIT_ZX[1] * stns_X
    np.save(PATH_OUT, stns_ZX)

    # H = np.copy(stns_ZX[0, :, :])  # fall height for f ONLY f!
    H = np.zeros((P.NUM_Z, P.NUM_X), dtype=np.uint16)  # fall height for f ONLY f!

    inds_buildup = np.where((BOUND_LO_y <= H_Z[:, :]) & (H_Z[:, :] <= BOUND_MI_y ))
    inds_break = np.where(BOUND_MI_y < H_Z[:, :])
    inds_post = np.where(H_Z[:, :] < BOUND_LO_y)

    H[inds_buildup] += int(100 * SPLIT_ZX[0])
    H[inds_break] += int(1000 * SPLIT_ZX[0])
    H[inds_post] += 0

    inds_buildup = np.where((BOUND_LO_y <= H_X[:, :]) & (H_X[:, :] <= BOUND_MI_y))
    inds_break = np.where(BOUND_MI_y < H_X[:, :])
    inds_post = np.where(H_X[:, :] < BOUND_LO_y)

    H[inds_buildup] += int(100 * SPLIT_ZX[1])
    H[inds_break] += int(1000 * SPLIT_ZX[1])
    H[inds_post] += 0

    H[np.where((H > 4) & (H < 500))] = 1
    H[np.where(H >= 500)] = 2

    return stns_ZX, H




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