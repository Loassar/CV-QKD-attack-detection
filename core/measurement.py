import numpy as np
import random
from scipy.special import erf
from scipy.optimize import minimize
from overrides import override
from core.constants import (
    VARIANCE_ALICE as VAR_0,
    SHOT_NOISE as N_0,
    ELECTRONIC_NOISE as N_el,
    TECHICAL_NOISE as N_tech,
    DETECTOR_EFFICIENCY as D_eff,
    CHANNEL_TRANSMITTANCE as T_ch,
    OVERALL_TRANSMISSION as T_ext,
    INTENSITY_LO as I_lo,
    MEAN_NORMAL,
)

class Measurement:
    """
    A parent class for modeling Bob's measurements in different cases (Unattacked, attack A, attack B and so on)

    NB! Scheme of modeling
    First, define init args(for example, num of samples) and constants like variance of Alice's distribution)
    Second, calculate params of Bob's distribution
    Finally, calculate Bob's distribution( quadratures)
    (args, constants) --> (variance, mean, shot noise, lo_intensity) --> Bob's quadratures
    """
    def __init__(self, attenuation_coefs: np.ndarray, samples: np.ndarray) -> None:
        """
        Parameters
        ----------
        attenuation_coefs : np.ndarray[r1, r2]
            An array of attenuation coefficients, where r1 is no attenuation, r2 is maximum attenuation.
        samples: np.ndarray[r1_samples, r2_samples]
            Number of measurements at different attenuation coefficients 
        """
        self.r = attenuation_coefs
        self.samples = samples

    def variance(self) -> np.ndarray:
        """
        Calculation of variance for modeling Bob's distribution. This method will be overridden for each measurement case.

        Returns
        -------
        np.ndarray[var_r1, var_r2]    
        Variance at different attenuation coefficients
        """
        pass

    def mean(self) -> np.ndarray:
        """
        Calculation of mean for modeling Bob's distribution. 

        Returns
        -------
        np.ndarray[mean_r1, mean_r2]    
        Mean at different attenuation coefficients.
        By default mean_r1 = mean_r2 = 0 for unattacked case
        """
        return np.array([MEAN_NORMAL, MEAN_NORMAL])  
    
    def quadrature_values(self) -> np.ndarray:
        """
        Calculation Bob's distribution.

        Generate random gauss distribution with params (mean, var, samples) -> 
            np.random.normal(mean, np.sqrt(var),n_samples) 
        For each attenuation coefficient -> 
            for var, n_samples, mean in zip(self.variance(), self.samples, self.mean()
        Combines into a single array -> 
            np.concatenate(...)

        Return
        ------
        np.ndarray[r1_samples + r2_samples]
        """
        return [
            np.random.normal(
                mean,  
                np.sqrt(var),
                n_samples
            ) for var, n_samples, mean in zip(self.variance(), self.samples, self.mean()) 
        ]


class Unattacked(Measurement):

    @override
    def variance(self) -> np.ndarray:
        """
        Formula: V_i = r_i * η * T * (V_a * N_0 + ξ) + N0 + V_el, where

        [N_0]->[N_0]    Shot noise variance
        [V_el]->[N_el]  Detector's electronic noise
        [ξ]->[N_tech]   Technical excess noise
        [T]->[T_ch]     Quantum channel transmittance
        [η]->[D_eff]    Efficiency of the homodyne detector
        [r_i]->[r]      Attenuation coefficients    
        """
        return  self.r * D_eff * T_ch * (VAR_0*N_0 + N_tech) + N_0 + N_el 


class LO_Intesity_Attack(Measurement):

    def __init__(self, attenuation_coefs: np.ndarray, samples: np.ndarray, k_loia: float) -> None:
        """
        In the LO intensity attack, Eve attacks the LO beam 
        by using a non-changing phase intensity attenuator with attenuation coefficient k(0 < k < 1).
        """
        super().__init__(attenuation_coefs, samples)
        self.k = k_loia

    @override
    def variance(self) -> np.ndarray:
        """
        Formulas: 
        (1) N = (1 - k * η * T) / (k * (1 - η * T)),

        (2) ξ_gau = (1 - η * T) * (N - 1) * N_0 / (η * T),

        (3) V_i = k * (r_i * η * T * (V_a * N_0 + ξ + ξ_gau) + N0 + V_el), where

        [N]->[VAR_epr]      Variance of Eve's EPR states
        [ξ_gau]->[N_gauss]  Noise introduced by Eve's Gaussian collective attack
        [N_0]->[N_0]        Shot noise variance
        [V_el]->[N_el]      Detector's electronic noise
        [ξ]->[N_tech]       Technical excess noise
        [T]->[T_ch]         Quantum channel transmittance
        [η]->[D_eff]        Efficiency of the homodyne detector
        [r_i]->[r]          Attenuation coefficients   
        """
        k = self.k
        # VAR_epr = (1 - k * D_eff * T_ch) / (k * (1 - D_eff * T_ch)) # Variance of Eve’s EPR states
        # N_gauss = (1 - D_eff * T_ch) * (VAR_epr - 1) * N_0 / (D_eff * T_ch) # Noise introduced by Eve’s Gaussian collective attack
        N_gauss = ((1 - k) * N_0) / (k * D_eff * T_ch)
        return k * (self.r * D_eff * T_ch * (VAR_0*N_0 + N_tech + N_gauss) + N_0 + N_el)
    

class Calibration_Attack(Measurement):
    
    def __init__(self, attenuation_coefs: np.ndarray, samples: np.ndarray, k_calib: float) -> None:
        """
         In the calibration attack, 
         Eve intercepts a fraction μ of the signal pulses by implementing a partial intercept-resend (PIR) attack 
         and modifies the shape of LO pulses to control the shot noise estimated by legitimate parties.
         [μ] -> k_calib
        """
        super().__init__(attenuation_coefs, samples)
        self.k = k_calib

    @override
    def variance(self) -> np.ndarray:
        """
        Formulas: 
        (1) ξ_pir = ξ + 2 * μ * N0,

        (2) ξ_calib = ξ_pir + (N0_calib - N0) / (η * T), 

        (3) V_i = r_i * η * T * N0_calib *  (V_a + ξ_calib + 2) + N0_calib + V_el * N0_calib, where

        [μ]->[k]                    Attenuation coefficient for calibration attack
        [ξ_pir]->[N_tech_pir]       Excess noise introduced by Eve's PIR attack
        [ξ_calib]->[N_tech_calib]   Excess noise introduced by calibration attack
        [N0]->[N_0]                 Shot noise variance
        [N0_calib]->[N_0_calib]     Shot noise after calibration attack
        [V_el]->[N_el]              Detector's electronic noise
        [ξ]->[N_tech]               Technical excess noise
        [T]->[T_ch]                 Quantum channel transmittance
        [η]->[D_eff]                Efficiency of the homodyne detector
        [r_i]->[r]                  Attenuation coefficients   
        """
        N_0_calib = N_0 / (1 + (2*self.k + 0.1) * D_eff * T_ch) # shot noise after calibration attack
        N_tech_pir = N_tech + 2 * self.k * N_0 # excess noise introduced by Eve's PIR attack
        N_tech_calib = N_tech_pir + (N_0_calib - N_0) / (D_eff * T_ch) # excess noise introduced by calibration attack
        return self.r * D_eff * T_ch * N_0_calib * (VAR_0 + N_tech_calib + 2) + N_0_calib + N_el * N_0_calib  
    
    
class Saturation_Attack(Measurement):
    
    def __init__(self, attenuation_coefs: np.ndarray, samples: np.ndarray, alpha_sat: float, delta: float) -> None:
        """
        In the saturation attack, Eve exploits the finite linearity domain of the homodyne detection response
        and intercepts all the pulses send by Alice and measures them with heterodyne detection, 
        then displaces the quadratures of the resent coherent states with a value Δ
        [Δ] -> delta
        [α] -> alpha is the boundary of the linear range of the homodyne detector

        """
        super().__init__(attenuation_coefs, samples)
        self.alpha = alpha_sat # boundary of the linear range of the homodyne detector
        self.delta = delta # displace of Bob's quadratures
        self.v_0 = self.r * D_eff * T_ch * (VAR_0*N_0 + N_tech + 2* N_0) + N_0 + N_el  # Bob Variance in linear range of the homodyne detector

    @override
    def mean(self) -> np.ndarray:
        """
        mean = r_i * (α + C),
        [Δ]->[delta]       Displace of Bob's quadratures
        [α]->[alpha]       Boundary of the linear range of the homodyne detector
        [C]->[compute_C]   Utility func
        """
        return self.r * (self.alpha + self.__compute_C())
    
    @override
    def variance(self) -> np.ndarray:
        A = self.__compute_A()
        B = self.__compute_B()
        alpha = self.alpha
        delta = self.delta
        v0 = self.v_0
        return (
            v0 * ((1 + A) / 2 - B**2 / (2 * np.pi)) -
            (alpha - delta) * np.sqrt(v0 / (2 * np.pi)) * A * B +
            (alpha - delta)**2 * (1 - A**2) / 4
        )
    
    def __compute_A(self):
        """
        Utility function for calculations
        TODO: Add formula to docstring
        """
        return erf((self.alpha - self.delta) / np.sqrt(2 * self.v_0))

    def __compute_B(self):
        """
        Utility function for calculations
        TODO: Add formula to docstring
        """
        return np.exp(-(self.alpha - self.delta)**2 / (2 * self.v_0))

    def __compute_C(self):
        """
        Utility function for calculations
        TODO: Add formula to docstring
        """
        return -(np.sqrt(self.v_0 / (2 * np.pi))*self.__compute_B() + (1 + self.__compute_A()) * (self.alpha - self.delta) / 2)


class Hybrid_Attack(Measurement):

    def __init__(self, attenuation_coefs: np.ndarray, samples: np.ndarray) -> None:
        super().__init__(attenuation_coefs, samples)


    @override
    def variance(self) -> np.ndarray:
        D, wavelength = self.find_params()
        return self.r * D_eff * T_ch * (VAR_0*N_0 + 2*N_0 + N_tech) + N_0 / wavelength + N_el + (1 - self.r) ** 2 * D ** 2 + (35.81 + 35.47 * self.r**2) * D
    

    def find_params(self):
        
        def constraint(vars):
            D, λ = vars
            return (N_0 / λ + (1 - self.r[0] * self.r[1]) * D**2 + (35.81 - 35.47 * self.r[0] * self.r[1]) * D) - N_0

        def equation(vars):
            D, λ = vars
            N_hyb = N_0 / λ + (1 - self.r[0] * self.r[1]) * D**2 + (35.81 - 35.47 * self.r[0] * self.r[1]) * D

            # Вычисляем ξ_hyb / N_hyb
            ξ_hyb_N_hyb = ((2 + N_tech) * N_0 + (self.r[0] + self.r[1] - 2) * D ** 2) / (D_eff * T_ch) + 35.47 * (self.r[0] + self.r[1]) * D

            # Минимизируем ξ_hyb / N_hyb
            return ξ_hyb_N_hyb / N_hyb
        
        
        x0 = np.array([1.0, 1.0])

        bounds = ((0, None), (0, None))

        con = {'type': 'eq', 'fun': constraint}

        # Минимизируем функцию objective с ограничением con и начальным приближением x0
        result = minimize(equation, x0, bounds=bounds, constraints=con)
        return result.x
    
class Hybrid_Attack2(Measurement):

    def __init__(self, attenuation_coefs: np.ndarray, samples: np.ndarray) -> None:
        super().__init__(attenuation_coefs, samples)

    @override
    def variance(self) -> np.ndarray:
        f_ext = 0.1
        I_ext = 11 * 10 ** 6
        R = I_ext / I_lo
        N_ext = 4 * T_ext * (1 - T_ext) * R / T_ch + R**2 * D_eff * f_ext**2 * (1 - 2*T_ext)**2 * I_lo / T_ch
        N_hyb2 = N_tech + 2 * N_0 + N_ext
        return self.r * D_eff * T_ch * (VAR_0*N_0 + N_hyb2) + N_0 + N_el 
    
    @override
    def mean(self) -> np.ndarray:
        I_ext = 11 * 10 ** 6
        mean = (D_eff/I_lo)**(1/2) * ( 1 - 2 * T_ext) * I_ext
        return np.array([mean, mean])