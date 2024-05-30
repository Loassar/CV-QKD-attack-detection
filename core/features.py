from core.measurement import Measurement
from core.input_state import InputState
import numpy as np
import random
from core.constants import (
    VARIANCE_ALICE as VAR_0,
    SHOT_NOISE as N_0,
    ELECTRONIC_NOISE as N_el,
    TECHICAL_NOISE as N_tech,
    DETECTOR_EFFICIENCY as D_eff,
    CHANNEL_TRANSMITTANCE as T_ch,
    INTENSITY_LO as I_lo,
    FLUCTUATION_PERCENT as F_per,
    ATTENUATION_VALUES as r,
    MEAN_NORMAL
)
class Features:
    """
    Class of Bob's distribution features, used to generate a set of feature vectors for further processing.
    """

    def __init__(self, blocks: np.ndarray, attack_coef: float = 1, kast:float = 1) -> None:
        self.blocks = blocks
        self.coef = attack_coef
        self.kast = kast


    def generate(self) -> np.ndarray:
        """
        Calculate [mean, variance, LO intensity, shot noise] for each block:
            [InputState(block.mean(), block.var(), self.intensity_lo(), self.shot_noise()).value 
                for block in blocks
        
        return 
        ------
        np.array[np.array[mean, var, intensity, noise]]
        """

        input = [InputState(np.mean(block[0]), np.var(block[0]), self.lo_intensity(self.coef), self.shot_noise(block[1])).value
            for block in self.blocks]
        return np.array(input)
    
    
    def shot_noise(self, block: np.ndarray) -> float:
        
        # return np.random.normal(N_0, N_0 * F_per)
       
        return (np.var(block) - N_el - N_tech * r[1] * D_eff * T_ch) / (1 + r[1] * D_eff * T_ch) 
    
    def lo_intensity(self, coef: float = 1) -> float:
        return np.random.normal(I_lo, I_lo * F_per) * coef