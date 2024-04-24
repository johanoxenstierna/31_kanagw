

import os
from scipy.stats._multivariate import multivariate_normal
import matplotlib.pyplot as plt
from src.trig_functions import _multivariate_normal
from matplotlib.pyplot import imread
import P


import numpy as np


def cut_k0(k0, inds_x, inds_z, d=None):
    """
    Generates the static pics
    """

    PATH_OUT = './pictures/k0_cut/'

    delete_old(PATH_OUT)

    '''Alpha mask'''
    # rv = multivariate_normal(mean=[d/2, d/2], cov=[[d*4, 0], [0, d*4]])  # more=more visible
    rv = multivariate_normal(mean=[d/2, d/2], cov=[[d*40000, 0], [0, d*40000]])  # more=more visible
    # x, y = np.mgrid[0:d*2:1, 0:d*2:1]
    x, y = np.mgrid[0:d:1, 0:d:1]
    pos = np.dstack((x, y))
    alpha_mask = rv.pdf(pos)
    alpha_mask = alpha_mask / np.max(alpha_mask)

    '''
    Obs this needs to correspond exactly with k0. 
    Needs to be flipped somehow. 
    plt.gca() is FLIPPED
    Caution this is fk up. 
    '''

    for i in range(len(inds_x)):
        for j in range(len(inds_z)):
            ind_x = inds_x[i]
            ind_z = inds_z[j]

            if ind_z < int(d/2):
                print("ind_z: " + str(ind_z), "   d: " + str(d))
                raise Exception("d too large")

            # pic = k0[ind_z + int(d/2):ind_z - int(d/2):-1, ind_x - int(d/2):ind_x + int(d/2), :]
            pic = k0[ind_z - int(d/2):ind_z + int(d/2), ind_x - int(d/2):ind_x + int(d/2), :]

            pic[:, :, 3] = alpha_mask

            pic_key = str(ind_x) + '_' + str(ind_z)
            np.save(PATH_OUT + pic_key, pic)


def delete_old(PATH):

    _, _, all_file_names = os.walk(PATH).__next__()

    removed_files = 0
    for file_name_rem in all_file_names:
        # print("removing " + str(file_name_rem))
        os.remove(PATH + file_name_rem)
        removed_files += 1
    print("removed_files: " + str(removed_files))


def get_c_d(k0, d):
    # c_ = k0[720:719 - d:-1, 100:100 + d, :]
    c_ = k0[720:719 - d:-1, 100:100 + d * 2, :]
    # d_ = k0[720:719 - d:-1, 0:d, :]
    d_ = k0[720:719 - d:-1, 0:d * 2, :]

    # rv = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * 3, 0], [0, d * 3]])  # less cov => less alpha, second one: width
    rv = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * 3, 0], [0, d * 6]])
    # x, y = np.mgrid[0:d*2:1, 0:d*2:1]
    x, y = np.mgrid[0:d:1, 0:d * 2:1]
    pos = np.dstack((x, y))
    mask = rv.pdf(pos)
    mask = mask / np.max(mask)

    c_[:, :, 3] = mask
    d_[:, :, 3] = mask

    return c_, d_


def get_kanagawa_fractals():

    """
    Need to write conversion from pixels to NUM_Z, NUM_X
    """

    PATH_IN = './pictures/waves/R/'
    _, _, all_file_names = os.walk(PATH_IN).__next__()

    R_ = {}
    R_inds_used = []

    ttt = imread(PATH_IN + '400_200' + '.png')
    ttt = np.flipud(ttt)
    R_['1_3'] = np.copy(ttt)
    R_['1_10'] = np.copy(ttt)
    R_['2_4'] = np.copy(ttt)
    R_['2_14'] = np.copy(ttt)
    R_['5_5'] = np.copy(ttt)
    R_['5_12'] = np.copy(ttt)

    R_inds_used.append((1, 3))  # z, x
    R_inds_used.append((1, 10))  # z, x
    R_inds_used.append((2, 4))  # z, x
    R_inds_used.append((2, 14))  # z, x
    R_inds_used.append((5, 5))  # z, x
    R_inds_used.append((5, 12))  # z, x


    return R_, R_inds_used

def convert_pixels_to_grid():
    pass




