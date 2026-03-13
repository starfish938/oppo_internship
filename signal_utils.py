import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def fft_analysis(signal, fs, title="Spectrum Analysis"):
    """
    FFT频谱分析并绘图
    """
    n = len(signal)
    freqs = fftfreq(n, 1/fs)[:n//2]
    fft_vals = fft(signal)[:n//2]
    magnitude = np.abs(fft_vals) / n * 2  

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, 20 * np.log10(magnitude + 1e-10))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return freqs, magnitude

def modulate(bits, mod_type='QPSK'):
    """
    通用调制函数,支持BPSK/QPSK/16QAM
    """
    mod_type = mod_type.upper()
    bits = np.asarray(bits)

    # 根据调制类型确定每符号比特数并截断输入
    if mod_type == 'BPSK':
        bits_per_sym = 1
    elif mod_type == 'QPSK':
        bits_per_sym = 2
    elif mod_type == '16QAM':
        bits_per_sym = 4
    else:
        raise ValueError(f"Unsupported modulation type: {mod_type}")

    # 截断到整数个符号
    n_symbols = len(bits) // bits_per_sym
    bits = bits[:n_symbols * bits_per_sym]
    symbols = np.zeros(n_symbols, dtype=complex)

    for i in range(n_symbols):
        sym_bits = bits[i*bits_per_sym : (i+1)*bits_per_sym]

        if mod_type == 'BPSK':
            # BPSK: 0 -> -1, 1 -> +1
            symbols[i] = 2 * sym_bits[0] - 1

        elif mod_type == 'QPSK':
            # QPSK: 比特对映射到 QPSK 星座点
            b1, b2 = sym_bits[0], sym_bits[1]
            if b1 == 0 and b2 == 0:
                symbols[i] = 1 + 1j
            elif b1 == 0 and b2 == 1:
                symbols[i] = -1 + 1j
            elif b1 == 1 and b2 == 0:
                symbols[i] = 1 - 1j
            else:
                symbols[i] = -1 - 1j

        else:  # 16QAM
            # 16QAM 映射表: 4比特 -> (I, Q)
            b1, b2, b3, b4 = sym_bits
            if (b1, b2) == (0, 0):
                I = -3
            elif (b1, b2) == (0, 1):
                I = -1
            elif (b1, b2) == (1, 0):
                I = 1
            else:  
                I = 3

            if (b3, b4) == (0, 0):
                Q = -3
            elif (b3, b4) == (0, 1):
                Q = -1
            elif (b3, b4) == (1, 0):
                Q = 1
            else: 
                Q = 3

            symbols[i] = I + 1j * Q

    # 能量归一化
    if mod_type == 'QPSK':
        symbols /= np.sqrt(2)
    elif mod_type == '16QAM':
        symbols /= np.sqrt(10)
    

    return symbols

def demodulate(symbols, mod_type='QPSK'):
    """
    通用解调函数
    """
    mod_type = mod_type.upper()
    symbols = np.asarray(symbols)
    bits = []

    if mod_type == 'BPSK':
        # BPSK: 实部 > 0 判为 1，否则 0
        for sym in symbols:
            bits.append(1 if sym.real > 0 else 0)

    elif mod_type == 'QPSK':
        symbols = symbols * np.sqrt(2)
        for sym in symbols:
            b1 = 0 if sym.real > 0 else 1
            b2 = 0 if sym.imag > 0 else 1
            bits.extend([b1, b2])

    else:  # 16QAM
        symbols = symbols * np.sqrt(10)
        for sym in symbols:
            I, Q = sym.real, sym.imag

            
            if I < -2:
                b1, b2 = 0, 0
            elif I < 0:
                b1, b2 = 0, 1
            elif I < 2:
                b1, b2 = 1, 0
            else:
                b1, b2 = 1, 1

            
            if Q < -2:
                b3, b4 = 0, 0
            elif Q < 0:
                b3, b4 = 0, 1
            elif Q < 2:
                b3, b4 = 1, 0
            else:
                b3, b4 = 1, 1

            bits.extend([b1, b2, b3, b4])

    return np.array(bits, dtype=int)

def add_awgn(signal, snr_db):
    """
    添加AWGN噪声
    """
    signal = np.asarray(signal)
    sig_pow = np.mean(np.abs(signal) ** 2)
    snr_lin = 10 ** (snr_db / 10)
    noise_pow = sig_pow / snr_lin

    # generate awgn
    if np.iscomplexobj(signal):
        noise = np.sqrt(noise_pow / 2) * (np.random.randn(*signal.shape) +
                                           1j * np.random.randn(*signal.shape))
    else:
        noise = np.sqrt(noise_pow) * np.random.randn(*signal.shape)

    return signal + noise

def calculate_ber(tx_bits, rx_bits):
    """
    计算误码率 (BER)
    """
    tx_bits = np.asarray(tx_bits)
    rx_bits = np.asarray(rx_bits)
    min_len = min(len(tx_bits), len(rx_bits))
    if min_len == 0:
        return 0.0
    errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
    return errors / min_len