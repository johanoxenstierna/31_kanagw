
import random
import P as P
import numpy as np

import P

# from sh_info import shInfoAbstract, _0_info
from O0s_info import _waves_info

def _genesis():

    '''
    Creates instance of each info and stores in dict
    '''

    gis = {}

    # if 'projectiles' in P.O0_TO_SHOW:
    #     projectiles_gi = _projectiles_info.Projectiles_info()
    #     gis['projectiles'] = projectiles_gi

    # p_waves = [5]
    if 'waves' in P.O0_TO_SHOW:  # EXPL
        waves_gi = _waves_info.Waves_info()
        gis['waves'] = waves_gi

    #
    # if 'clouds' in P.O0_TO_SHOW:  # EXPL
    #     clouds_gi = _clouds_info.Clouds_info()
    #     gis['clouds'] = clouds_gi


    return gis