

import os
from scipy.stats._multivariate import multivariate_normal
import matplotlib.pyplot as plt
from src.trig_functions import _multivariate_normal
from matplotlib.pyplot import imread
import P


import numpy as np


def prep_k0(inds_x, inds_z, d):
    """
    Generates the static pics
    """

    PATH_OUT = './pictures/k0_cut/'

    delete_old(PATH_OUT)

    k0 = imread('./pictures/k0.png')

    w, h = 20, 20  # width, height
    # d = 10  # diameter

    '''Alpha mask'''
    rv = multivariate_normal(mean=[d, d], cov=[[d*20, 0], [0, d*20]])
    x, y = np.mgrid[0:d*2:1, 0:d*2:1]
    pos = np.dstack((x, y))
    mask = rv.pdf(pos)
    mask = mask / np.max(mask)

    '''Obs this needs to correspond exactly with k0'''

    for i in range(len(inds_x)):
        for j in range(len(inds_z)):
            ind_x = inds_x[i]
            ind_z = inds_z[j]

            # aa = k0[:, 60:20:-1, :]

            pic = k0[ind_z + d:ind_z - d:-1, ind_x - d:ind_x + d, :]

            alpha = pic[:, :, 3] * mask
            # pic[:, :, 3] = alpha

            pic_key = str(ind_x) + '_' + str(ind_z)
            np.save(PATH_OUT + pic_key, pic)


def delete_old(PATH):

    _, _, all_file_names = os.walk(PATH).__next__()

    removed_files = 0
    for file_name_rem in all_file_names:
        print("removing " + str(file_name_rem))
        os.remove(PATH + file_name_rem)
        removed_files += 1
    print("removed_files: " + str(removed_files))
