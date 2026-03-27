import numpy as np
import matplotlib.pyplot as plt

# LS信道估计
def ls_channel_estimation(received_pilots, pilot_values):
    H_ls_pilots = received_pilots / pilot_values
    return H_ls_pilots


# 线性插值
def linear_interpolation(H_pilots, pilot_indices, n_subcarriers):
    all_indices = np.arange(n_subcarriers)

    H_real = np.interp(all_indices, pilot_indices, np.real(H_pilots))
    H_imag = np.interp(all_indices, pilot_indices, np.imag(H_pilots))

    return H_real + 1j * H_imag


# AWGN
def add_awgn(signal, snr_db):
    snr = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / snr

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )

    return signal + noise


# OFDM调制
def ofdm_modulate(symbols, N_fft, N_cp):
    time_signal = np.fft.ifft(symbols, N_fft)
    cp = time_signal[-N_cp:]
    return np.concatenate([cp, time_signal])


# OFDM解调
def ofdm_demodulate(rx_signal, N_fft, N_cp):
    rx_signal = rx_signal[N_cp:N_cp + N_fft]
    return np.fft.fft(rx_signal, N_fft)


# QPSK调制
def modulate(bits):
    bits = bits.reshape(-1, 2)
    symbols = (2 * bits[:,0] - 1) + 1j * (2 * bits[:,1] - 1)
    return symbols / np.sqrt(2)


# QPSK解调
def demodulate(symbols):
    bits = np.zeros((len(symbols),2))
    bits[:,0] = np.real(symbols) > 0
    bits[:,1] = np.imag(symbols) > 0
    return bits.flatten()


# BER
def calculate_ber(tx_bits, rx_bits):
    return np.mean(tx_bits != rx_bits)


# NMSE
def calculate_nmse(H_true, H_estimated):

    error = H_true - H_estimated

    nmse = np.mean(np.abs(error)**2) / np.mean(np.abs(H_true)**2)

    return 10 * np.log10(nmse)



# OFDM仿真（使用LS信道估计）
def ofdm_simulation_with_ls(snr_db, channel_h_time):

    N_fft = 64
    N_cp = 16
    N_subcarriers = 64

    pilot_indices = np.arange(0,64,8)
    data_indices = np.setdiff1d(np.arange(64), pilot_indices)

    # ===== 发送端 =====
    tx_bits = np.random.randint(0,2,len(data_indices)*2)

    data_symbols = modulate(tx_bits)

    tx_grid = np.zeros(N_subcarriers, dtype=complex)

    tx_grid[data_indices] = data_symbols
    tx_grid[pilot_indices] = 1 + 0j

    tx_signal = ofdm_modulate(tx_grid, N_fft, N_cp)


    # ===== 信道 =====
    rx_signal = np.convolve(tx_signal, channel_h_time)[:len(tx_signal)]

    rx_signal = add_awgn(rx_signal, snr_db)


    # ===== 接收端 =====
    rx_grid = ofdm_demodulate(rx_signal, N_fft, N_cp)


    # LS信道估计
    rx_pilots = rx_grid[pilot_indices]

    pilot_values_tx = np.ones(len(pilot_indices), dtype=complex)

    H_ls_pilots = ls_channel_estimation(rx_pilots, pilot_values_tx)

    H_ls_all = linear_interpolation(H_ls_pilots, pilot_indices, N_subcarriers)


    # 均衡
    H_ls_safe = H_ls_all.copy()
    H_ls_safe[np.abs(H_ls_safe) <= 1e-6] = 1e-6

    rx_equalized = rx_grid / H_ls_safe


    # 解调
    rx_data_symbols = rx_equalized[data_indices]

    rx_bits = demodulate(rx_data_symbols)

    ber = calculate_ber(tx_bits, rx_bits)


    # NMSE
    H_true = np.fft.fft(channel_h_time, N_fft)[:N_subcarriers]

    nmse = calculate_nmse(H_true, H_ls_all)

    return ber, nmse



# BER曲线
snr_range = np.arange(0,26,2)

channel_h = np.array([0.8+0j,0.4+0.2j,0.2+0.1j])

ber_ls_list = []

for snr in snr_range:

    ber_vals = [ofdm_simulation_with_ls(snr, channel_h)[0] for _ in range(100)]

    ber_ls_list.append(np.mean(ber_vals))


plt.figure(figsize=(8,6))

plt.semilogy(snr_range, ber_ls_list, 'b-o', label='LS ', linewidth=2)

plt.xlabel('SNR (dB)')
plt.ylabel('BER')

plt.title('OFDM  BER (LS channe estimate)')

plt.legend()

plt.grid(True, which='both', alpha=0.3)

plt.show()