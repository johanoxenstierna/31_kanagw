

import os
import random

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
    Use H instead
    """

    PATH_IN = './pictures/waves/R/'
    _, _, all_file_names = os.walk(PATH_IN).__next__()
    NUM_REPEATS_P_PIC = 25

    d = np.zeros(shape=(P.NUM_Z, P.NUM_X))
    indices = list(np.ndindex(d.shape))
    np.random.shuffle(indices) # total random

    R_ = {}
    R_inds_used = []

    for file_name in all_file_names:

        ttt = imread(PATH_IN + file_name)
        ttt = np.flipud(ttt)
        possible_cells = convert_pixels_to_possible_cells(file_name.split('.')[0])
        random.shuffle(possible_cells)

        for _ in range(NUM_REPEATS_P_PIC):

            ind_zx = possible_cells.pop()
            ind_zx_str = str(ind_zx[0]) + '_' + str(ind_zx[1])

            if ind_zx not in R_inds_used:
                R_inds_used.append(ind_zx)
                R_[ind_zx_str] = np.copy(ttt)


    # R_['1_3'] = np.copy(ttt)  # zx
    # R_['1_10'] = np.copy(ttt)
    # R_['2_4'] = np.copy(ttt)
    # R_['2_14'] = np.copy(ttt)
    # R_['5_5'] = np.copy(ttt)
    # R_['15_20'] = np.copy(ttt)
    # R_['15_40'] = np.copy(ttt)
    # R_['20_5'] = np.copy(ttt)
    # R_['20_50'] = np.copy(ttt)
    # R_['25_10'] = np.copy(ttt)

    # R_inds_used.append((1, 3))  # z, x
    # R_inds_used.append((1, 10))  # z, x
    # R_inds_used.append((2, 4))  # z, x
    # R_inds_used.append((2, 14))  # z, x
    # R_inds_used.append((5, 5))  # z, x
    # R_inds_used.append((15, 20))  # z, x
    # R_inds_used.append((15, 40))  # z, x
    # R_inds_used.append((20, 5))  # z, x
    # R_inds_used.append((20, 50))  # z, x
    # R_inds_used.append((25, 10))  # z, x

    return R_, R_inds_used

def convert_pixels_to_possible_cells(pixels):

    cells = []

    '''Get ratio'''
    pxsl_x = int(pixels.split('_')[0])
    pxsl_z = int(pixels.split('_')[1])

    ratio_x = pxsl_x / 1280
    ratio_z = (720 - pxsl_z) / 720  # FROM BOTTOM
    
    ratio_x_diff = 0.3
    ratio_z_diff = 0.3

    cell_x_min = int((ratio_x - ratio_x_diff) * P.NUM_X)
    cell_x_max = int((ratio_x + ratio_x_diff) * P.NUM_X)

    cell_z_min = int((ratio_z - ratio_z_diff) * P.NUM_Z)
    cell_z_max = int((ratio_z + ratio_z_diff) * P.NUM_Z)

    for i in range(cell_x_min, cell_x_max):
        for j in range(cell_z_min, cell_z_max):
            cells.append((i, j))

    return cells