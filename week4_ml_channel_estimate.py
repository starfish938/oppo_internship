"""
week4_ml_channel_estimation.py
Mini Project 4: MMSE + MLP Channel Estimation
Compare NMSE performance of LS, MMSE and MLP estimators
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import interp1d


# ===================== Channel Model =====================

def generate_rayleigh_channel(n_paths=3, max_delay_samples=8, n_subcarriers=52, n_fft=64):
    """
    Generate a multipath Rayleigh fading channel.

    Returns
    -------
    h_time : time-domain impulse response
    H_freq : frequency response on active subcarriers
    """

    delays = np.sort(np.random.randint(0, max_delay_samples, n_paths))
    gains = (np.random.randn(n_paths) + 1j * np.random.randn(n_paths)) / np.sqrt(2)

    h_time = np.zeros(n_fft, dtype=complex)

    for d, g in zip(delays, gains):
        h_time[d] += g

    h_time = h_time / np.sqrt(np.sum(np.abs(h_time) ** 2))

    H_full = np.fft.fft(h_time, n_fft)

    start = (n_fft - n_subcarriers) // 2
    H_freq = H_full[start:start + n_subcarriers]

    return h_time, H_freq


# ===================== Utility Functions =====================

def calculate_nmse(H_true, H_est):
    """Compute NMSE in dB"""
    mse = np.mean(np.abs(H_true - H_est) ** 2)
    power = np.mean(np.abs(H_true) ** 2)
    return 10 * np.log10(mse / power)


def complex_to_real(H_complex):
    """Convert complex vector to real representation"""
    return np.concatenate([np.real(H_complex), np.imag(H_complex)])


def real_to_complex(H_real):
    """Convert real vector back to complex"""
    n = len(H_real) // 2
    return H_real[:n] + 1j * H_real[n:]


# ===================== MMSE Estimator =====================

def mmse_channel_estimation(H_ls_pilots, snr_db, R_HH=None):

    n_pilots = len(H_ls_pilots)

    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1.0 / snr_linear

    if R_HH is None:
        R_HH = np.eye(n_pilots, dtype=complex)

    W = R_HH @ np.linalg.inv(R_HH + noise_var * np.eye(n_pilots))

    H_mmse = W @ H_ls_pilots

    return H_mmse


# ===================== Neural Network =====================

class SimpleMLP(nn.Module):

    def __init__(self, n_pilots, n_subcarriers):
        super(SimpleMLP, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(n_pilots * 2, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, n_subcarriers * 2)

        )

    def forward(self, x):
        return self.network(x)


# ===================== Dataset Generation =====================

def generate_dataset(n_samples, snr_range_db, n_pilots, n_subcarriers, pilot_indices):

    X_data = []
    Y_data = []

    for _ in range(n_samples):

        _, H_true = generate_rayleigh_channel(
            n_paths=3,
            max_delay_samples=8,
            n_subcarriers=n_subcarriers,
            n_fft=64
        )

        snr_db = np.random.uniform(snr_range_db[0], snr_range_db[1])

        H_pilots_true = H_true[pilot_indices]

        noise_var = 1.0 / (10 ** (snr_db / 10))

        noise = np.sqrt(noise_var / 2) * (
                np.random.randn(n_pilots) +
                1j * np.random.randn(n_pilots)
        )

        H_ls_pilots = H_pilots_true + noise

        X = complex_to_real(H_ls_pilots)
        Y = complex_to_real(H_true)

        X_data.append(X)
        Y_data.append(Y)

    return np.array(X_data, dtype=np.float32), np.array(Y_data, dtype=np.float32)


# ===================== Training =====================

def train_mlp(model, train_loader, val_X, val_Y, n_epochs=50, lr=0.001):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

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
            val_pred = model(val_X)
            val_loss = criterion(val_pred, val_Y).item()

        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}  Train Loss: {train_losses[-1]:.4f}  Val Loss: {val_loss:.4f}")

    plt.figure(figsize=(10, 4))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MLP Training Curve")

    plt.legend()
    plt.grid(True)
    plt.show()

    return train_losses, val_losses


# ===================== Evaluation =====================

def evaluate_nmse(model, snr_db, n_test, pilot_indices, n_subcarriers=52):

    nmse_list = []

    for _ in range(n_test):

        _, H_true = generate_rayleigh_channel(3, 8, n_subcarriers, 64)

        H_pilots_true = H_true[pilot_indices]

        noise_var = 1.0 / (10 ** (snr_db / 10))

        noise = np.sqrt(noise_var / 2) * (
                np.random.randn(len(pilot_indices)) +
                1j * np.random.randn(len(pilot_indices))
        )

        H_ls_pilots = H_pilots_true + noise

        X_input = torch.FloatTensor(
            complex_to_real(H_ls_pilots)
        ).unsqueeze(0)

        model.eval()

        with torch.no_grad():
            Y_pred = model(X_input).numpy()[0]

        H_mlp = real_to_complex(Y_pred)

        nmse = calculate_nmse(H_true, H_mlp)

        nmse_list.append(nmse)

    return np.mean(nmse_list)


def evaluate_ls_mmse(snr_db, n_test, pilot_indices, n_subcarriers=52):

    nmse_ls_list = []
    nmse_mmse_list = []

    all_subcarriers = np.arange(n_subcarriers)

    for _ in range(n_test):

        _, H_true = generate_rayleigh_channel(3, 8, n_subcarriers, 64)

        H_pilots_true = H_true[pilot_indices]

        noise_var = 1.0 / (10 ** (snr_db / 10))

        noise = np.sqrt(noise_var / 2) * (
                np.random.randn(len(pilot_indices)) +
                1j * np.random.randn(len(pilot_indices))
        )

        H_ls_pilots = H_pilots_true + noise

        interp_ls = interp1d(
            pilot_indices,
            H_ls_pilots,
            kind='linear',
            fill_value="extrapolate"
        )

        H_ls_full = interp_ls(all_subcarriers)

        H_mmse_pilots = mmse_channel_estimation(
            H_ls_pilots,
            snr_db
        )

        interp_mmse = interp1d(
            pilot_indices,
            H_mmse_pilots,
            kind='linear',
            fill_value="extrapolate"
        )

        H_mmse_full = interp_mmse(all_subcarriers)

        nmse_ls_list.append(
            calculate_nmse(H_true, H_ls_full)
        )

        nmse_mmse_list.append(
            calculate_nmse(H_true, H_mmse_full)
        )

    return np.mean(nmse_ls_list), np.mean(nmse_mmse_list)


# ===================== Main Program =====================

if __name__ == "__main__":

    n_pilots = 13
    n_subcarriers = 52

    pilot_indices = np.linspace(
        0,
        n_subcarriers - 1,
        n_pilots,
        dtype=int
    )

    print("Generating dataset...")

    X_train, Y_train = generate_dataset(
        8000,
        [0, 25],
        n_pilots,
        n_subcarriers,
        pilot_indices
    )

    X_val, Y_val = generate_dataset(
        1000,
        [0, 25],
        n_pilots,
        n_subcarriers,
        pilot_indices
    )

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")

    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.FloatTensor(Y_train)

    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.FloatTensor(Y_val)

    train_dataset = TensorDataset(X_train_t, Y_train_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    model = SimpleMLP(
        n_pilots=n_pilots,
        n_subcarriers=n_subcarriers
    )

    print("Start training MLP...")

    train_mlp(
        model,
        train_loader,
        X_val_t,
        Y_val_t,
        n_epochs=50,
        lr=0.001
    )

    torch.save(
        model.state_dict(),
        "mlp_channel_estimator.pth"
    )

    print("Model saved: mlp_channel_estimator.pth")

    snr_range = np.arange(0, 26, 5)

    nmse_ls = []
    nmse_mmse = []
    nmse_mlp = []

    print("Evaluating estimators...")

    for snr in snr_range:

        ls_val, mmse_val = evaluate_ls_mmse(
            snr,
            200,
            pilot_indices,
            n_subcarriers
        )

        nmse_ls.append(ls_val)
        nmse_mmse.append(mmse_val)

        mlp_val = evaluate_nmse(
            model,
            snr,
            200,
            pilot_indices,
            n_subcarriers
        )

        nmse_mlp.append(mlp_val)

        print(
            f"SNR={snr}dB  LS={ls_val:.2f}dB  MMSE={mmse_val:.2f}dB  MLP={mlp_val:.2f}dB"
        )

    plt.figure(figsize=(8, 6))

    plt.plot(snr_range, nmse_ls, 'b--o', label="LS")
    plt.plot(snr_range, nmse_mmse, 'g--s', label="MMSE")
    plt.plot(snr_range, nmse_mlp, 'r-s', label="MLP")

    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("Channel Estimation Comparison")

    plt.legend()
    plt.grid(True)
    plt.show()

    print("Mini Project 4 finished.")