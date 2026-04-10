import numpy as np

def mmse_channel_estimation(H_ls_pilots, snr_db, pilot_indices, n_subcarriers, delay_spread=8):
    """
    基于导频位置的MMSE估计 (仅针对导频点)
    H_ls_pilots: 导频位置上的LS估计 (复数)
    snr_db: 当前信噪比(dB)
    pilot_indices: 导频子载波索引
    n_subcarriers: 总子载波数
    delay_spread: 均方根时延扩展 (采样点)
    返回: 导频位置上MMSE估计后的信道
    """
    n_pilots = len(pilot_indices)
    # 计算频域相关矩阵 (指数衰减功率延迟谱)
    # 子载波间隔归一化为1，相关系数 rho = 1/(1 + j*2*pi*delta_k*delay_spread/N_fft)
    R_pp = np.zeros((n_pilots, n_pilots), dtype=complex)
    for i in range(n_pilots):
        for j in range(n_pilots):
            delta_k = pilot_indices[i] - pilot_indices[j]
            # 简化模型: 相关系数随频率差指数衰减
            R_pp[i, j] = np.exp(-0.5 * (delta_k * delay_spread / n_subcarriers)**2)
    
    # 噪声方差
    noise_var = 1.0 / (10**(snr_db/10))
    # MMSE 滤波矩阵: R_pp * (R_pp + noise_var * I)^{-1}
    reg_matrix = R_pp + noise_var * np.eye(n_pilots)
    W = np.linalg.solve(reg_matrix, R_pp).T   # 或者直接 np.dot(R_pp, np.linalg.inv(reg_matrix))
    H_mmse_pilots = np.dot(W, H_ls_pilots)
    return H_mmse_pilots