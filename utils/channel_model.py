import numpy as np

def generate_rayleigh_channel(n_taps, max_delay, n_subcarriers, n_fft):
    """
    生成瑞利衰落信道的频域响应
    n_taps: 多径数目
    max_delay: 最大时延 (采样点数)
    n_subcarriers: 使用的子载波数
    n_fft: FFT大小
    返回: (时域冲激响应 h, 频域响应 H_true)
    """
    # 时域抽头: 复高斯随机变量，功率指数衰减
    delays = np.linspace(0, max_delay, n_taps)
    powers = np.exp(-delays / (max_delay/2))
    powers = powers / np.sum(powers)
    h = np.zeros(n_fft, dtype=complex)
    for i in range(n_taps):
        tap_amp = np.sqrt(powers[i]/2) * (np.random.randn() + 1j*np.random.randn())
        h[int(delays[i])] = tap_amp
    H = np.fft.fft(h, n_fft)
    H_true = H[:n_subcarriers]   # 只取使用的子载波
    return h, H_true