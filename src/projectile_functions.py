import copy
import random
import numpy as np


def simple_projectile(gi=None):
    """
    OBS this is for midpoint, i.e. SINGLE PIXEL
    See tutorial for using a patch to make it larger than single pixel
    always assumes origin is at 0, 0, so needs shifting afterwards.
    ALSO, THIS FUNCTION ALWAYS OUTPUTS RIGHT MOTION, UNTIL IT IS SET BELOW
    """

    '''HERE TUNE THETA BASED ON V. THETA CLOSE TO 0 -> MORE V'''

    xy = np.zeros((gi['frames_tot'], 2))  # MIDPOINT

    G = 9.8
    # h = gi['v'] * 10  # THIS IS THE NUMBER OF PIXELS IT WILL GO
    h = 0.7 * 400 + 0.3 * gi['v'] * 5  # THIS IS THE NUMBER OF PIXELS IT WILL GO

    # t_flight = 6 * v * np.sin(theta) / G

    '''
    OBS since projectile is launched from a height, the calculation is different:
    https://www.omnicalculator.com/physics/time-of-flight-projectile-motion
    from ground level: 
    t_flight = 4 * gi['v'] * np.sin(gi['theta']) / G  # 4 means they land at origin. 5 little bit below
    
    '''
    t_flight = (gi['v'] * np.sin(gi['theta']) + np.sqrt((gi['v'] * np.sin(gi['theta']))**2 + 2 * G * h)) / G

    t_lin = np.linspace(0, t_flight, gi['frames_tot'])
    # t_geo = np.geomspace(0.08, t_flight ** 1.2, gi['frames_tot'])
    # t_geo_0 = np.geomspace(0.5, t_flight ** 1, gi['frames_tot'])  # POWER CONTROLS DISTANCE
    # t_geo_1 = np.geomspace(0.5, t_flight ** 1, gi['frames_tot'])

    x = gi['v'] * np.cos(gi['theta']) * t_lin
    # x_lin = abs(gi['v'] * np.cos(gi['theta']) * t_lin)  # THIS IS ALWAYS POSITIVE
    # x_geo = abs(2 * gi['v'] * np.cos(gi['theta']) * t_geo_0)  # THIS IS ALWAYS POSITIVE. KEEP IT SIMPLE
    # # x = 0.0001 * x_lin * t_lin + 0.2 * x_lin * x_geo
    # # x = 0.00001 * x_lin * t_lin + 0.1 * x_lin * x_geo
    # # x = 0.001 * x_lin * t_lin + 0.005 * x_lin * x_geo
    # # x = 0.05 * x_lin + 0.95 * x_geo
    # x = x

    '''If theta is close enough '''
    y = gi['v'] * np.sin(gi['theta']) * 2 * t_lin - 0.5 * G * t_lin ** 2

    # y_lin = gi['v'] * np.sin(gi['theta']) * 2 * t_lin #- 0.5 * G * t_lin ** 2  # OBS OBS this affect both up and down equally
    # y_geo = gi['v'] * np.sin(gi['theta']) * 2 * t_geo_1 - 0.5 * G * t_geo_1 ** 2


    xy[:, 0] = x
    xy[:, 1] = y

    return xy


def flip_projectile_x(sp):
    """Only works for sp"""

    # # PEND DEL
    # if sp.gi['ld_init'][0] < sp.f.gi['left_mid']:  # flip x values
    #     sp.xy_t[:, 0] = -sp.xy_t[:, 0]
    #
    # # let a random subset still go over middle
    # dist_to_mid = abs(sp.gi['ld'][0] - 640)
    # if dist_to_mid < 50 and sp.gi['dist_to_theta_0'] < 0.4:
    #     if random.random() < 0.4:
    #         if sp.gi['ld'][0] < sp.f.gi['left_mid']:
    #             sp.xy_t[:, 0] = -sp.xy_t[:, 0]
    # elif dist_to_mid < 100 and sp.gi['dist_to_theta_0'] < 0.2:
    #     if random.random() < 0.2:
    #         if sp.gi['ld'][0] < sp.f.gi['left_mid']:
    #             sp.xy_t[:, 0] = -sp.xy_t[:, 0]

    if random.random() < 0.5:
        sp.xy_t[:, 0] = -sp.xy_t[:, 0]

    return sp.xy_t


def flip_and_shift_object(xy_t, origin=None, gi=None):
    """
    OBS N6 = its hardcoded for sp
    shifts it to desired xy
    y is flipped because 0 y is at top and if flip_it=True
    """

    xy = copy.deepcopy(xy_t)

    # '''THIS MAKES THEM NEVER CROSS MIDDLE'''
    # if origin[0] < 640 and xy[0, 0] < xy[-1, 0]:  # TO THE LEFT BUT POSITIVE
    #     xy[:, 0] = -xy[:, 0]
    # elif origin[0] > 640 and xy[0, 0] > xy[-1, 0]:  # TO THE RIGHT BUT NEGATIVE
    #     xy[:, 0] = -xy[:, 0]
    #
    # # '''let a random subset still go over middle'''
    # dist_to_mid = abs(origin[0] - 640)
    # if dist_to_mid < 50 and gi['dist_to_theta_0'] < 0.4:
    #     if random.random() < 0.4:
    #         if origin[0] < 640:
    #             xy[:, 0] = -xy[:, 0]
    # elif dist_to_mid < 100 and gi['dist_to_theta_0'] < 0.2:
    #     if random.random() < 0.2:
    #         if origin[0] < 640:
    #             xy[:, 0] = -xy[:, 0]

    # if gi['out_screen']:
    #     xy = xy[:int(len(xy) / 2), :int(len(xy) / 2)]
    #     gi['frames_tot'] = len(xy)

    '''TODO: Check whether values in array are not what they're supposed to be and flip.'''
    # if gi['up_down'] == 'up' or gi['up_down'] == 'down':
    #     xy[:, 1] *= -1  # flip it. Now all neg

    '''x'''
    xy[:, 0] += origin[0]  # OBS THIS ORIGIN MAY BE BOTH LEFT AND RIGHT OF 640

    '''
    y1: Move. y_shift_r_f_d is MORE shifting downward (i.e. positive), but only the latter portion 
    of frames is shown.
    '''
    xy[:, 1] += origin[1] #- xy[0, 1]

    return xy




