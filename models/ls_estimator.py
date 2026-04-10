import numpy as np

def ls_channel_estimation(rx_pilots, tx_pilots):
    """
    最小二乘 (LS) 信道估计
    rx_pilots: 接收到的导频符号
    tx_pilots: 发送的导频符号 (已知)
    """
    return rx_pilots / tx_pilots

def linear_interpolation(estimated_pilots, pilot_indices, total_subcarriers):
    """
    对导频位置的估计值进行线性插值，得到全子载波信道估计
    estimated_pilots: 复数数组，导频位置上的估计值
    pilot_indices: 导频所在子载波索引
    total_subcarriers: 子载波总数
    """
    H_full = np.zeros(total_subcarriers, dtype=complex)
    # 对实部和虚部分别插值
    real_pilots = np.real(estimated_pilots)
    imag_pilots = np.imag(estimated_pilots)
    all_idx = np.arange(total_subcarriers)
    real_interp = np.interp(all_idx, pilot_indices, real_pilots)
    imag_interp = np.interp(all_idx, pilot_indices, imag_pilots)
    H_full = real_interp + 1j * imag_interp
    return H_full