from dataclasses import dataclass
import numpy as np


@dataclass
class InputState:
    mean_v: float
    variance: float
    lo_intensity: float
    shot_noise: float

    @property
    def value(self) -> np.ndarray:
        """Возвращает состояние как вектор np.ndarray формы 
        [mean_v, variance, lo_intensity, shot_noise]

        Returns
        -------
        np.ndarray
            [mean_v, variance, lo_intensity, shot_noise]
        """
        return np.array([self.mean_v, self.variance,
                         self.lo_intensity, self.shot_noise])
