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

# 参数配置
config = {

    "mod_type": "QPSK",      # BPSK / QPSK / 16QAM
    "channel_type": "3path", # AWGN / 2path / 3path
    "snr_range": np.arange(0,25,2)

}


# 调制

def modulate(bits, mod_type='QPSK'):

    if mod_type == 'BPSK':

        symbols = 2*bits - 1
        return symbols.astype(complex)

    elif mod_type == 'QPSK':

        bits = bits.reshape((-1,2))

        symbols = (2*bits[:,0]-1) + 1j*(2*bits[:,1]-1)

        symbols = symbols/np.sqrt(2)

        return symbols

    elif mod_type == '16QAM':

        bits = bits.reshape((-1,4))

        real = (2*bits[:,0]-1)*(2-bits[:,2])
        imag = (2*bits[:,1]-1)*(2-bits[:,3])

        symbols = real + 1j*imag

        symbols = symbols/np.sqrt(10)

        return symbols


# 解调

def demodulate(symbols, mod_type='QPSK'):

    bits = []

    if mod_type == 'BPSK':

        for s in symbols:
            bits.append(1 if s.real > 0 else 0)

    elif mod_type == 'QPSK':

        for s in symbols:

            bits.append(1 if s.real > 0 else 0)
            bits.append(1 if s.imag > 0 else 0)

    elif mod_type == '16QAM':

        for s in symbols:

            bits.append(1 if s.real > 0 else 0)
            bits.append(1 if s.imag > 0 else 0)

            bits.append(0 if abs(s.real) > 1 else 1)
            bits.append(0 if abs(s.imag) > 1 else 1)

    return np.array(bits)

#插入导频

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



# 信道

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



# BER

def calculate_ber(tx_bits, rx_bits):

    errors = np.sum(tx_bits != rx_bits)

    return errors / len(tx_bits)



# 信道类型

def get_channel(channel_type):

    if channel_type == "AWGN":
        return [1]

    elif channel_type == "2path":
        return [0.8,0.4]

    elif channel_type == "3path":
        return [0.8,0.4,0.2]


# OFDM仿真

def ofdm_simulation(snr_db, channel_taps, mod_type='QPSK'):

    if mod_type == 'BPSK':
        n_bits = len(data_indices)

    elif mod_type == 'QPSK':
        n_bits = len(data_indices)*2

    elif mod_type == '16QAM':
        n_bits = len(data_indices)*4

    tx_bits = np.random.randint(0,2,n_bits)

    tx_symbols = modulate(tx_bits, mod_type)

    resource_grid = insert_pilots(tx_symbols)

    tx_signal = ofdm_modulate(resource_grid, N_fft, N_cp)

    rx_signal = multipath_channel(tx_signal, channel_taps, snr_db)

    rx_grid = ofdm_demodulate(rx_signal, N_fft, N_cp)

    H_true = np.fft.fft(channel_taps, N_fft)[:N_subcarriers]

    rx_equalized = rx_grid / H_true

    rx_data_symbols = rx_equalized[data_indices]

    rx_bits = demodulate(rx_data_symbols, mod_type)

    ber = calculate_ber(tx_bits, rx_bits)

    return ber, tx_symbols, rx_data_symbols



# 资源网格图

def plot_resource_grid():

    grid = np.zeros((N_subcarriers, N_symbols))

    for s in range(N_symbols):

        grid[pilot_indices,s] = 1
        grid[data_indices,s] = 2

    plt.imshow(grid, aspect='auto')

    plt.xlabel("OFDM Symbol (Time)")
    plt.ylabel("Subcarrier(Frequency)")

    plt.title("Resource Grid")

    plt.colorbar(label="1=pilot,2=data")



# 主程序

channel_taps = get_channel(config["channel_type"])

snr_range = config["snr_range"]

ber_results = []

for snr in snr_range:

    ber, tx_symbols, rx_symbols = ofdm_simulation(
        snr,
        channel_taps,
        config["mod_type"]
    )

    ber_results.append(ber)


plt.figure(figsize=(10,8))

# 发送星座图
plt.subplot(2,2,1)
plt.scatter(tx_symbols.real, tx_symbols.imag)
plt.title("TX Constellation")
plt.grid(True)


# 接收星座图
plt.subplot(2,2,2)
plt.scatter(rx_symbols.real, rx_symbols.imag)
plt.title("RX Constellation")
plt.grid(True)


# BER曲线
plt.subplot(2,2,3)
plt.semilogy(snr_range, ber_results, marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BER vs SNR")
plt.grid(True)


# 资源网格
plt.subplot(2,2,4)
plot_resource_grid()


plt.tight_layout()
plt.show()