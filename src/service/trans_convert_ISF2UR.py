import numpy as np
from copy import deepcopy

def trans_convert_ISF2UR(translation_XYZ_ISF):
    # ISF座標系からUR座標系への変換
    translation_XYZ_UR = np.zeros(3)
    # ---
    translation_XYZ_UR[0] = deepcopy(translation_XYZ_ISF[1])
    translation_XYZ_UR[1] = deepcopy(-translation_XYZ_ISF[0])
    translation_XYZ_UR[2] = deepcopy(translation_XYZ_ISF[2])
    # ---
    return translation_XYZ_UR
