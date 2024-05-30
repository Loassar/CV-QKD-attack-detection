VARIANCE_ALICE = 10 # Variance of Alice quadratures

SHOT_NOISE = 0.4 # [N_0] Shot noise variance
ELECTRONIC_NOISE = 0.01 * SHOT_NOISE # [V_el] Detector's electronic noise
TECHICAL_NOISE = 0.1 * SHOT_NOISE # [ξ] Technical excess noise

ALPHA = 0.2 # [α] Loss coefficient of the optical fiber
TRANSMISSION_DISTANCE = 30 # Transmission distance

CHANNEL_TRANSMITTANCE = 10 ** (-ALPHA * TRANSMISSION_DISTANCE/10) # Quantum channel transmittance
DETECTOR_EFFICIENCY = 0.6 # [η] Efficiency of the homodyne detector

INTENSITY_LO = 10 ** 7 # LO Intensity

NUM_SAMPLES = 1 * 10 ** 7 # Data size of each attack
NUM_BLOCKS = 1 * 10 ** 3 # Each block contains Q = 10 ** 4 pulses, so NUM_BLOCKS = NUM_SAMPLES / Q = 10 ** 3

ATTENUATION_VALUES = [1, 0.001] # [no, max] attenuations

FLUCTUATION_PERCENT = 0.01 # fluctuation of intensity/shot noise
K = 0.95 # attenuation coefficient

MEAN_NORMAL = 0 # mean without attack   

OVERALL_TRANSMISSION = 0.51