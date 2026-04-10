"""
Week 5 Mini Project 5: CNN Channel Estimation + System Integration
One-click run all experiments, generate comparison charts and result files
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# Import custom modules
from models.ls_estimator import ls_channel_estimation, linear_interpolation
from models.mmse_estimator import mmse_channel_estimation
from models.cnn_estimator import ChannelNet
from utils.channel_model import generate_rayleigh_channel
from utils.metrics import calculate_nmse

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ========== Global Parameters ==========
N_SUBCARRIERS = 52
PILOT_INDICES = np.arange(0, N_SUBCARRIERS, 6)  # 9 pilots uniformly inserted
N_FFT = 64
CHANNEL_TAPS = 3
MAX_DELAY = 8

# ========== 1. Generate CNN Dataset (Day22) ==========
def generate_cnn_dataset(n_samples, snr_range_db, n_subcarriers, pilot_indices):
    """
    Generate CNN dataset for channel estimation
    
    Input format: [2, N_subcarriers] (2 channels: real + imag)
    - Pilot positions: fill with LS estimates
    - Data positions: fill with 0 (to be estimated)
    Label format: [2, N_subcarriers] (true channel)
    """
    X_data, Y_data = [], []
    for _ in range(n_samples):
        _, H_true = generate_rayleigh_channel(CHANNEL_TAPS, MAX_DELAY, n_subcarriers, N_FFT)
        snr_db = np.random.uniform(snr_range_db[0], snr_range_db[1])
        
        # LS estimation (pilot positions)
        H_pilots_true = H_true[pilot_indices]
        noise_var = 1.0 / (10**(snr_db/10))
        noise = np.sqrt(noise_var/2) * (np.random.randn(len(pilot_indices)) + 1j*np.random.randn(len(pilot_indices)))
        H_ls_pilots = H_pilots_true + noise
        
        # Build input: sparse LS estimates (pilot positions have values, others are 0)
        H_input = np.zeros(n_subcarriers, dtype=complex)
        H_input[pilot_indices] = H_ls_pilots
        
        # Convert to 2-channel format [2, N_sc]
        X = np.stack([np.real(H_input), np.imag(H_input)], axis=0)
        Y = np.stack([np.real(H_true), np.imag(H_true)], axis=0)
        
        X_data.append(X)
        Y_data.append(Y)
    
    return np.array(X_data, dtype=np.float32), np.array(Y_data, dtype=np.float32)

# ========== 2. Train CNN Model (Day22) ==========
def train_cnn_model():
    print("Generating CNN training dataset...")
    X_train, Y_train = generate_cnn_dataset(8000, [0, 25], N_SUBCARRIERS, PILOT_INDICES)
    X_val, Y_val   = generate_cnn_dataset(1000, [0, 25], N_SUBCARRIERS, PILOT_INDICES)
    
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)
    X_val_t   = torch.FloatTensor(X_val)
    Y_val_t   = torch.FloatTensor(Y_val)
    
    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=64, shuffle=True)
    
    model = ChannelNet(n_subcarriers=N_SUBCARRIERS, n_filters=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    n_epochs = 50
    train_losses, val_losses = [], []
    
    print("Start training CNN (ChannelNet)...")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()
        
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "cnn_channel_estimator.pth")
    print("CNN model saved as cnn_channel_estimator.pth")
    
    # Plot training curve (Figure 4)
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('CNN (ChannelNet) Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/cnn_training_curve.png', dpi=150)
    plt.show()
    
    return model

# ========== 3. Full Performance Comparison (Day23-24) ==========
def run_full_comparison(snr_range, model_cnn, n_test=200):
    """
    Run complete performance comparison: LS + linear interpolation vs MMSE vs CNN
    Returns dictionary with NMSE and BER results
    """
    results = {
        'snr': snr_range,
        'nmse_ls': [], 'nmse_mmse': [], 'nmse_cnn': [],
        'ber_ls': [], 'ber_mmse': [], 'ber_cnn': [], 'ber_ideal': []
    }
    
    for snr_db in snr_range:
        print(f"Testing SNR = {snr_db} dB...")
        nmse_ls_list, nmse_mmse_list, nmse_cnn_list = [], [], []
        ber_ls_list, ber_mmse_list, ber_cnn_list, ber_ideal_list = [], [], [], []
        
        for _ in range(n_test):
            # Generate true channel
            _, H_true = generate_rayleigh_channel(CHANNEL_TAPS, MAX_DELAY, N_SUBCARRIERS, N_FFT)
            
            # Pilot reception with noise
            H_pilots_true = H_true[PILOT_INDICES]
            noise_var = 1.0 / (10**(snr_db/10))
            noise = np.sqrt(noise_var/2) * (np.random.randn(len(PILOT_INDICES)) + 1j*np.random.randn(len(PILOT_INDICES)))
            rx_pilots = H_pilots_true + noise
            
            # ----- LS estimation -----
            H_ls_pilots = ls_channel_estimation(rx_pilots, np.ones(len(PILOT_INDICES), dtype=complex))
            H_ls_all = linear_interpolation(H_ls_pilots, PILOT_INDICES, N_SUBCARRIERS)
            nmse_ls_list.append(calculate_nmse(H_true, H_ls_all))
            
            # ----- MMSE estimation (MMSE filtering on pilots + linear interpolation) -----
            H_mmse_pilots = mmse_channel_estimation(H_ls_pilots, snr_db, PILOT_INDICES, N_SUBCARRIERS)
            H_mmse_all = linear_interpolation(H_mmse_pilots, PILOT_INDICES, N_SUBCARRIERS)
            nmse_mmse_list.append(calculate_nmse(H_true, H_mmse_all))
            
            # ----- CNN estimation -----
            H_input = np.zeros(N_SUBCARRIERS, dtype=complex)
            H_input[PILOT_INDICES] = H_ls_pilots
            X_cnn = torch.FloatTensor(np.stack([np.real(H_input), np.imag(H_input)], axis=0)).unsqueeze(0)
            model_cnn.eval()
            with torch.no_grad():
                Y_cnn = model_cnn(X_cnn).numpy()[0]   # [2, N_sc]
            H_cnn = Y_cnn[0] + 1j * Y_cnn[1]
            nmse_cnn_list.append(calculate_nmse(H_true, H_cnn))
            
            # ----- BER approximation (simplified, following project plan formula) -----
            for H_est, ber_list in [(H_true, ber_ideal_list), (H_ls_all, ber_ls_list),
                                     (H_mmse_all, ber_mmse_list), (H_cnn, ber_cnn_list)]:
                nmse_val = calculate_nmse(H_true, H_est)
                ber_approx = max(0.5 * np.exp(nmse_val / 5), 1e-5)
                ber_list.append(ber_approx)
        
        results['nmse_ls'].append(np.mean(nmse_ls_list))
        results['nmse_mmse'].append(np.mean(nmse_mmse_list))
        results['nmse_cnn'].append(np.mean(nmse_cnn_list))
        results['ber_ideal'].append(np.mean(ber_ideal_list))
        results['ber_ls'].append(np.mean(ber_ls_list))
        results['ber_mmse'].append(np.mean(ber_mmse_list))
        results['ber_cnn'].append(np.mean(ber_cnn_list))
    
    return results

# ========== 4. Visualization: Channel Estimation Example (Figure 3) ==========
def plot_channel_example(snr_db=15, model_cnn=None):
    """At given SNR, show magnitude comparison of true channel and three estimation methods"""
    _, H_true = generate_rayleigh_channel(CHANNEL_TAPS, MAX_DELAY, N_SUBCARRIERS, N_FFT)
    H_pilots_true = H_true[PILOT_INDICES]
    noise_var = 1.0 / (10**(snr_db/10))
    noise = np.sqrt(noise_var/2) * (np.random.randn(len(PILOT_INDICES)) + 1j*np.random.randn(len(PILOT_INDICES)))
    rx_pilots = H_pilots_true + noise
    
    H_ls_pilots = ls_channel_estimation(rx_pilots, np.ones(len(PILOT_INDICES), dtype=complex))
    H_ls_all = linear_interpolation(H_ls_pilots, PILOT_INDICES, N_SUBCARRIERS)
    
    H_mmse_pilots = mmse_channel_estimation(H_ls_pilots, snr_db, PILOT_INDICES, N_SUBCARRIERS)
    H_mmse_all = linear_interpolation(H_mmse_pilots, PILOT_INDICES, N_SUBCARRIERS)
    
    H_input = np.zeros(N_SUBCARRIERS, dtype=complex)
    H_input[PILOT_INDICES] = H_ls_pilots
    X_cnn = torch.FloatTensor(np.stack([np.real(H_input), np.imag(H_input)], axis=0)).unsqueeze(0)
    model_cnn.eval()
    with torch.no_grad():
        Y_cnn = model_cnn(X_cnn).numpy()[0]
    H_cnn = Y_cnn[0] + 1j * Y_cnn[1]
    
    plt.figure(figsize=(12, 6))
    subcarriers = np.arange(N_SUBCARRIERS)
    plt.plot(subcarriers, np.abs(H_true), 'k-', linewidth=2, label='True Channel')
    plt.plot(subcarriers, np.abs(H_ls_all), 'b--o', markersize=4, label='LS + Interpolation')
    plt.plot(subcarriers, np.abs(H_mmse_all), 'g--^', markersize=4, label='MMSE + Interpolation')
    plt.plot(subcarriers, np.abs(H_cnn), 'r-s', markersize=4, label='CNN (ChannelNet)')
    plt.scatter(PILOT_INDICES, np.abs(H_true[PILOT_INDICES]), c='k', marker='x', s=80, label='Pilot Positions')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('|H| (Channel Magnitude)')
    plt.title(f'Channel Estimation Visualization (SNR = {snr_db} dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/channel_estimation_visual.png', dpi=150)
    plt.show()

# ========== 5. Main Flow ==========
def main():
    # Create results folder
    os.makedirs('results', exist_ok=True)
    
    # Train CNN model (and save training curve)
    model_cnn = train_cnn_model()
    
    # Run performance comparison experiment
    snr_range = np.arange(0, 26, 5)
    results = run_full_comparison(snr_range, model_cnn, n_test=200)
    
    # Save CSV results
    df = pd.DataFrame({
        'SNR_dB': results['snr'],
        'NMSE_LS_dB': results['nmse_ls'],
        'NMSE_MMSE_dB': results['nmse_mmse'],
        'NMSE_CNN_dB': results['nmse_cnn'],
        'BER_LS': results['ber_ls'],
        'BER_MMSE': results['ber_mmse'],
        'BER_CNN': results['ber_cnn'],
        'BER_Ideal': results['ber_ideal']
    })
    df.to_csv('results/comparison_results.csv', index=False)
    print("Experiment results saved to results/comparison_results.csv")
    
    # Figure 1: NMSE Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(results['snr'], results['nmse_ls'], 'b--o', label='LS + Linear Interpolation', linewidth=2)
    plt.plot(results['snr'], results['nmse_mmse'], 'g--^', label='MMSE', linewidth=2)
    plt.plot(results['snr'], results['nmse_cnn'], 'r-s', label='CNN (ChannelNet)', linewidth=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (dB)')
    plt.title('Channel Estimation NMSE Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/nmse_comparison.png', dpi=150)
    plt.show()
    
    # Figure 2: BER Comparison
    plt.figure(figsize=(10, 6))
    plt.semilogy(results['snr'], results['ber_ideal'], 'k-', label='Ideal Channel Estimation (Upper Bound)', linewidth=2)
    plt.semilogy(results['snr'], results['ber_ls'], 'b--o', label='LS + Linear Interpolation', linewidth=2)
    plt.semilogy(results['snr'], results['ber_mmse'], 'g--^', label='MMSE', linewidth=2)
    plt.semilogy(results['snr'], results['ber_cnn'], 'r-s', label='CNN (ChannelNet)', linewidth=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('OFDM System BER Performance Comparison (QPSK Modulation)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/ber_comparison.png', dpi=150)
    plt.show()
    
    # Figure 3: Channel estimation visualization example (SNR=15dB)
    plot_channel_example(snr_db=15, model_cnn=model_cnn)
    
    print("Mini Project 5 Complete! All results saved in results/ directory.")

if __name__ == "__main__":
    main()