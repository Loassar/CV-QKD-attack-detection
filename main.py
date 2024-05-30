import numpy as np
import math
from core.features import Features

from core.measurement import (
    Unattacked, 
    LO_Intesity_Attack,
    Calibration_Attack,
    Saturation_Attack
    )
from core.constants import (
    ATTENUATION_VALUES,
    NUM_SAMPLES,
    NUM_BLOCKS,
    SHOT_NOISE as N0,
)

def run():

    attenuation_coefs = np.array(ATTENUATION_VALUES)
    samples = np.array([int(NUM_SAMPLES * 0.9), int(NUM_SAMPLES * 0.1)])

    # --------------------------------------------------------------------------------------------------

    unattacked_m = Unattacked(attenuation_coefs=attenuation_coefs, samples=samples)
    unattacked_data = Features(unattacked_m).generate(NUM_BLOCKS)

    # --------------------------------------------------------------------------------------------------

    k_loia = 0.95

    loia_m = LO_Intesity_Attack(attenuation_coefs=attenuation_coefs, samples=samples, k_loia=k_loia)
    loia_data = Features(loia_m).generate(NUM_BLOCKS)
    
    # --------------------------------------------------------------------------------------------------

    k_calib = 1

    calib_m = Calibration_Attack(attenuation_coefs=attenuation_coefs, samples=samples, k_calib=k_calib)
    calib_data = Features(calib_m).generate(NUM_BLOCKS)


    # --------------------------------------------------------------------------------------------------

    a_sat = 20 * math.sqrt(N0)
    delta_sat = 19.5 * math.sqrt(N0)

    sat_m = Saturation_Attack(attenuation_coefs=attenuation_coefs, samples=samples, alpha_sat=a_sat, delta=delta_sat)
    sat_data = Features(sat_m).generate(NUM_BLOCKS)
   # --------------------------------------------------------------------------------------------------
    
    print(f'{unattacked_data.shape}, {loia_data.shape}, {calib_data.shape}, {sat_data.shape}')


if __name__ ==  '__main__':
    run()