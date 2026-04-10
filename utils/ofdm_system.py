import numpy as np
import matplotlib.pyplot as plt

# 系统参数
N_fft = 64
N_cp = 8
N_subcarriers = 52
N_symbols = 14
pilot_spacing = 4

pilot_indices = np.arange(0, N_subcarriers, pilot_spacing)
data_indices = np.setdiff1d(np.arange(N_subcarriers), pilot_indices)

# QPSK调制
def modulate(bits, mod_type='QPSK'):

    if mod_type == 'QPSK':

        bits = bits.reshape((-1,2))

        symbols = (2*bits[:,0]-1) + 1j*(2*bits[:,1]-1)

        symbols = symbols/np.sqrt(2)

        return symbols

# QPSK解调

def demodulate(symbols, mod_type='QPSK'):

    bits = []

    for s in symbols:

        bits.append(1 if s.real > 0 else 0)
        bits.append(1 if s.imag > 0 else 0)

    return np.array(bits)

# 插入导频
def insert_pilots(data_symbols, pilot_value=1+0j):

    resource_grid = np.zeros(N_subcarriers, dtype=complex)

    resource_grid[pilot_indices] = pilot_value

    resource_grid[data_indices] = data_symbols

    return resource_grid

# OFDM调制
def ofdm_modulate(resource_grid, N_fft, N_cp):

    freq_domain = np.zeros(N_fft, dtype=complex)

    freq_domain[:N_subcarriers] = resource_grid

    time_domain = np.fft.ifft(freq_domain)

    cp = time_domain[-N_cp:]

    tx_signal = np.concatenate([cp, time_domain])

    return tx_signal


# OFDM解调
def ofdm_demodulate(rx_signal, N_fft, N_cp):

    rx_signal = rx_signal[N_cp:]

    freq_domain = np.fft.fft(rx_signal)

    return freq_domain[:N_subcarriers]

# 多径信道 + 噪声

def multipath_channel(tx_signal, channel_taps, snr_db):

    channel = np.array(channel_taps)

    rx_signal = np.convolve(tx_signal, channel)[:len(tx_signal)]

    signal_power = np.mean(np.abs(rx_signal)**2)

    snr_linear = 10**(snr_db/10)

    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power/2) * (
        np.random.randn(len(rx_signal)) +
        1j*np.random.randn(len(rx_signal))
    )

    return rx_signal + noise

# BER计算

def calculate_ber(tx_bits, rx_bits):

    errors = np.sum(tx_bits != rx_bits)

    return errors / len(tx_bits)



# 完整OFDM仿真

def ofdm_simulation(snr_db, channel_taps, mod_type='QPSK'):

    # 1 生成比特
    n_bits = len(data_indices) * 2

    tx_bits = np.random.randint(0,2,n_bits)

    # 2 调制
    tx_symbols = modulate(tx_bits, mod_type)

    # 3 插入导频
    resource_grid = insert_pilots(tx_symbols)

    # 4 OFDM调制
    tx_signal = ofdm_modulate(resource_grid, N_fft, N_cp)

    # 5 信道
    rx_signal = multipath_channel(tx_signal, channel_taps, snr_db)

    # 6 OFDM解调
    rx_grid = ofdm_demodulate(rx_signal, N_fft, N_cp)

    # 7 理想信道估计
    H_true = np.fft.fft(channel_taps, N_fft)[:N_subcarriers]

    # 8 均衡
    rx_equalized = rx_grid / H_true

    # 9 提取数据
    rx_data_symbols = rx_equalized[data_indices]

    rx_bits = demodulate(rx_data_symbols, mod_type)

    # 10 BER
    ber = calculate_ber(tx_bits, rx_bits)

    return ber


# 运行仿真

snr_range = np.arange(0,25,2)

ber_results = []

for snr in snr_range:

    ber = ofdm_simulation(snr, channel_taps=[0.8,0.4,0.2])

    ber_results.append(ber)


# 画BER曲线

plt.semilogy(snr_range, ber_results, marker='o')

plt.xlabel("SNR (dB)")
plt.ylabel("BER")

plt.title("OFDM BER vs SNR")

plt.grid(True)

plt.show()