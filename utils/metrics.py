import numpy as np

def calculate_nmse(H_true, H_est):
    """
    计算归一化均方误差 (NMSE) 单位: dB
    H_true: 真实信道 (复数)
    H_est:  估计信道 (复数)
    """
    mse = np.mean(np.abs(H_true - H_est)**2)
    power = np.mean(np.abs(H_true)**2)
    nmse = 10 * np.log10(mse / (power + 1e-8))
    return nmse