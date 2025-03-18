import h5py
import numpy as np
import matplotlib.pyplot as plt
import itertools
import argparse
import pandas as pd
import sys
import os

# Get the absolute path of the modules directory (one level up + modules)
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules"))

# Add to Python path
sys.path.append(module_path)


import impulsivity as impul
import calculate_snr_rpr as csr

# Argument Parser for Event Selection
parser = argparse.ArgumentParser(description="Process an event from an HDF5 file")
parser.add_argument("--h5_file", type=str, required=True, help="Path to the HDF5 file")
parser.add_argument("--event_number", type=int, required=True, help="Event number (0-499)")
args = parser.parse_args()

# Load HDF5 File
with h5py.File(args.h5_file, "r") as f:
    waveforms = f["waveform_collect"][:]  # Shape (16, 500, 1230)

# Select the event
event_number = args.event_number
if event_number < 0 or event_number >= 500:
    raise ValueError(f"Invalid event number {event_number}, must be in range 0-499")

waveform_event = waveforms[:, event_number, :]  # Shape (16, 1230)

# Generate a time axis (assuming 0.5 ns per sample)
num_samples = waveform_event.shape[1]
time = np.linspace(0, num_samples * 0.5, num_samples)  # 0.5 ns per sample

# Calculate SNR and RPR for each channel
p2p_values = []
rms_values = []
snr_values = []
rpr_values = []
smoothed_waveforms = []
max_rpr_indices = []
sum_win_sizes = []

for ch in range(16):
    ch_waveform = waveform_event[ch, :]

    # Calculate P2P, RMS, SNR
    p2p, best_start_idx, best_end_idx = csr.calculate_p2p(ch_waveform)
    rms = csr.calculate_rms(ch_waveform)
    snr = p2p / (2 * rms)

    # Calculate RPR
    rpr, max_rpr_idx, smoothed_wf, sum_win = csr.calculate_rpr(ch_waveform, time)

    # Store Values
    p2p_values.append(p2p)
    rms_values.append(rms)
    snr_values.append(snr)
    rpr_values.append(rpr)
    smoothed_waveforms.append(smoothed_wf)
    max_rpr_indices.append(max_rpr_idx)
    sum_win_sizes.append(sum_win)

# Plot 16-Channel Waveforms in a 4×4 Grid
fig, axes = plt.subplots(4, 4, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ch in range(16):
    axes[ch].plot(time, waveform_event[ch, :], label=f"Channel {ch}")
    axes[ch].set_title(f"Channel {ch}")
    axes[ch].grid(True, linestyle="--", alpha=0.5)
    axes[ch].legend(fontsize=8)

plt.suptitle(f"Waveforms for Event {event_number}", fontsize=14)
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig(f"waveforms_event_{event_number}.png")
plt.show()

# Plot P2P, RMS, and SNR Results
plt.figure(figsize=(8, 6))
plt.bar(range(16), p2p_values, label="P2P", alpha=0.7, color="blue")
plt.bar(range(16), rms_values, label="RMS", alpha=0.7, color="orange")
plt.bar(range(16), snr_values, label="SNR", alpha=0.7, color="green")
plt.xlabel("Channel Number")
plt.ylabel("Value")
plt.title(f"P2P, RMS, SNR for Event {event_number}")
plt.legend()
plt.grid()
plt.savefig(f"snr_rms_p2p_event_{event_number}.png")
plt.show()

# Plot RPR Values
plt.figure(figsize=(8, 6))
plt.bar(range(16), rpr_values, label="RPR", alpha=0.7, color="red")
plt.xlabel("Channel Number")
plt.ylabel("RPR Value")
plt.title(f"RPR for Event {event_number}")
plt.legend()
plt.grid()
plt.savefig(f"rpr_event_{event_number}.png")
plt.show()

# Plot Waveform, P2P, and RPR for Channel 0
ch = 0
highlight_start = max(0, max_rpr_indices[ch] - sum_win_sizes[ch] // 2)
highlight_end = min(len(time) - 1, max_rpr_indices[ch] + sum_win_sizes[ch] // 2)

plt.figure(figsize=(10, 5))
plt.plot(time, waveform_event[ch, :], label="Original Waveform", color="blue", alpha=0.8)
plt.plot(time, smoothed_waveforms[ch], label="Smoothed Waveform", color="red", linewidth=2)

for i in range(highlight_start, highlight_end):
    plt.plot(time[i:i+2], smoothed_waveforms[ch][i:i+2], color="black", linewidth=2)

plt.title(f"Smoothed Waveform (RPR calculation example)")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(f"rpr_channel_{ch}_event_{event_number}.png")
plt.show()

# Compute Impulsivity
impulsivity, cdf = impul.get_impulsivity(waveform_event[0, :])
n_imp, n_cdf = impul.get_impulsivity(waveform_event[1, :])  # Compare with another channel

slope, intercept, t, cdf_fit = impul.get_cdf_fit(time, cdf)
slope1, intercept1, t, cdf_noise_fit = impul.get_cdf_fit(time, n_cdf)

plt.figure(figsize=(8, 6))
plt.plot(t, cdf, label="Signal CDF", color="b")
plt.plot(t, n_cdf, label="Noise CDF", color="orange")
plt.axhline(np.mean(cdf), label="Avg Signal CDF", linestyle="--", color="blue")
plt.axhline(np.mean(n_cdf), label="Avg Noise CDF", linestyle="--", color="orange")
plt.legend()
plt.xlabel("Normalized Time")
plt.ylabel("Normalized CDF")
plt.grid()
plt.savefig(f"impulsivity_cdf_event_{event_number}.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t, cdf, label="Signal CDF", color="b")
plt.plot(t, cdf_fit, label="Signal CDF Fit", color="b", linestyle="--")
plt.plot(t, n_cdf, label="Noise CDF", color="orange")
plt.plot(t, cdf_noise_fit, label="Noise CDF Fit", color="orange", linestyle="--")
plt.xlabel("Normalized time")
plt.ylabel("Normalized CDF")
plt.legend()
plt.savefig(f"impulsivity_fit_event_{event_number}.png")
plt.show()

print(f"Saved all plots for event {event_number}")

