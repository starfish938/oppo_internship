import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.gridspec as gridspec
import os

def generate_signal(sig_type = 'sine', fs = 1000, duration = 1.0, f1 = 50, f2 = 200):
    """
    生成测试信号
    """
    t = np.arange(0, duration, 1/fs)

    if sig_type == 'sine':
        signal_data = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    elif sig_type == 'square':
        signal_data = signal.square(2 * np.pi * f1 * t)
    elif sig_type == 'chirp':
        signal_data = signal.chirp(t, f0=f1, t1=duration, f1=f2, method='linear')
    
    else:
        raise ValueError(f"Value Error:{sig_type}")
    
    return t[:len(signal_data)], signal_data

def apply_window(signal_data, window_type = "hamming"):
    """
    应用窗函数
    """
    N = len(signal_data)

    if window_type == 'rect':
        window = np.ones(N)
    elif window_type == 'hamming':
        window = np.hamming(N)
    else:
        raise ValueError(f"Windows Error: {window_type}")
    
    windowed_signal = signal_data * window

    return windowed_signal, window

def compute_spectrum(signal_data, fs, window_type = 'hamming'):
    """
    计算信号的频谱
    """

    windowed, _ = apply_window(signal_data, window_type)

    #fft
    N = len(windowed)
    fft_vals = np.fft.fft(windowed)
    freqs = np.fft.fftfreq(N, 1/fs)

    half = N // 2
    freqs = freqs[:half]
    magnitude = np.abs(fft_vals[:half]) * 2 / N

    return freqs, magnitude

def plot_spectrum_analyzer(signal_data, fs, window_type = 'hamming', sig_name = 'Signal', nperseg = 256):
    """
    绘制频谱分析仪
    """

    t = np.arange(len(signal_data)) / fs
    
    # calculate
    freqs, magnitude = compute_spectrum(signal_data, fs, window_type)
    
    # STFT
    f_stft, t_stft, Sxx = signal.spectrogram(signal_data, fs, 
                                             window=window_type,
                                             nperseg=nperseg, 
                                             noverlap=nperseg//2)
    
    #创建图形布局
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.3, wspace=0.25)
    
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    plot_len = min(500, len(signal_data))
    ax1.plot(t[:plot_len], signal_data[:plot_len], 'b-', linewidth=1.5)
    ax1.set_title(f'Time Domain Waveform - {sig_name}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([t[0], t[plot_len-1]])
    
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.stem(freqs, magnitude, linefmt='g-', markerfmt='go', basefmt='r-', 
             label=f'{window_type} window')
    ax2.set_title('Frequency Spectrum', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)', fontsize=10)
    ax2.set_ylabel('Magnitude', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, fs/2])
    ax2.legend()
    
    
    ax3 = fig.add_subplot(gs[1, :])
    # 转换为dB刻度
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    im = ax3.pcolormesh(t_stft, f_stft, Sxx_db, shading='gouraud', cmap='jet')
    ax3.set_title('Time-Frequency Analysis (STFT)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Frequency (Hz)', fontsize=10)
    ax3.set_ylim([0, fs/2])
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Power (dB)', fontsize=10)
    
    
    fig.suptitle(f'Spectrum Analyzer v1.0 - Window: {window_type}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig
1

def interactive_mode():
    """
    交互模式：让用户选择信号类型和窗函数
    """
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    
    # 参数设置
    fs = 1000
    duration = 2.0
    
    # 用户选择信号类型
    print("\nPlease choose a signal:")
    print("1. sine (50Hz + 200Hz)")
    print("2. square (50Hz)")
    print("3. Chirp (10Hz → 300Hz)")
    print("4. random ")
    
    choice = input("please enter a number (1-4): ")
    
    sig_map = {
        '1': {'type': 'sine', 'name': 'Sine Wave', 'f1': 50, 'f2': 200},
        '2': {'type': 'square', 'name': 'Square Wave', 'f1': 50, 'f2': 150},
        '3': {'type': 'chirp', 'name': 'Chirp Signal', 'f1': 10, 'f2': 300},
        '4': {'type': 'voice', 'name': 'Noise', 'f1': 100, 'f2': 400}
    }
    
    if choice not in sig_map:
        print("Invalid choice---use sine wave")
        sig_info = sig_map['1']
    else:
        sig_info = sig_map[choice]
    
    # 用户选择窗函数
    print("\nplease choose window:")
    print("1. (Rectangular)")
    print("2. (Hamming)")
    
    win_choice = input("please enter numbers (1-2): ")
    window_type = 'rect' if win_choice == '1' else 'hamming'
    
    # 生成信号
    print(f"\ngenerate {sig_info['name']}...")
    t, signal_data = generate_signal(
        sig_type=sig_info['type'],
        fs=fs,
        duration=duration,
        f1=sig_info['f1'],
        f2=sig_info['f2']
    )
    
    # 分析并显示
    print("Spectrum analysising...")
    fig = plot_spectrum_analyzer(
        signal_data, fs,
        window_type=window_type,
        sig_name=sig_info['name'],
        nperseg=256
    )
    
    plt.show()
    print("\nFinish!")

if __name__ == "__main__":
    interactive_mode()