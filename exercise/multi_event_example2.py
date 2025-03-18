import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse

import sys
import os

# Get the absolute path of the modules directory (one level up + modules)
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules"))
# Add to Python path
sys.path.append(module_path)

import calculate_snr_rpr as csr
import impulsivity as impul
from scipy.stats import linregress

# Argument Parser for HDF5 File Input
parser = argparse.ArgumentParser(description="Process all events from an HDF5 file")
parser.add_argument("--h5_file", type=str, required=True, help="Path to the HDF5 file")
args = parser.parse_args()

# Load HDF5 File
with h5py.File(args.h5_file, "r") as f:
    waveforms = f["waveform_collect"][:]  # Shape (16, 500, 1230)

num_channels, num_events, num_samples = waveforms.shape

# Generate time axis (assuming 0.5 ns per sample)
time = np.linspace(0, num_samples * 0.5, num_samples)  # 0.5 ns per sample

# Storage for parameters
p2p_values = np.zeros((num_channels, num_events))
rms_values = np.zeros((num_channels, num_events))
snr_values = np.zeros((num_channels, num_events))
rpr_values = np.zeros((num_channels, num_events))
impulsivity_values = np.zeros((num_channels, num_events))
slope_values = np.zeros((num_channels, num_events))
intercept_values = np.zeros((num_channels, num_events))

# Storage for CSW
p2p_csw_values = np.zeros(num_events)
rms_csw_values = np.zeros(num_events)
snr_csw_values = np.zeros(num_events)
rpr_csw_values = np.zeros(num_events)
impulsivity_csw_values = np.zeros(num_events)
slope_csw_values = np.zeros(num_events)
intercept_csw_values = np.zeros(num_events)

# Process each event
for event in range(num_events):
    waveform_event = waveforms[:, event, :]  # Shape (16, 1230)
    
    for ch in range(num_channels):
        ch_waveform = waveform_event[ch, :]

        # Calculate P2P, RMS, SNR
        p2p, _, _ = csr.calculate_p2p(ch_waveform)
        rms = csr.calculate_rms(ch_waveform)
        snr = p2p / (2 * rms)

        # Calculate RPR
        rpr, _, _, _ = csr.calculate_rpr(ch_waveform, time)

        # Compute Impulsivity
        impulsivity, cdf = impul.get_impulsivity(ch_waveform)
        slope, intercept, _, _ = impul.get_cdf_fit(time, cdf)
        if impulsivity > 0.2:
           print('event',event)
        # Store Values
        p2p_values[ch, event] = p2p
        rms_values[ch, event] = rms
        snr_values[ch, event] = snr
        rpr_values[ch, event] = rpr
        impulsivity_values[ch, event] = impulsivity
        slope_values[ch, event] = slope
        intercept_values[ch, event] = intercept

    # Compute CSW for the first 8 channels
    csw = np.sum(waveform_event[:8, :], axis=0) / 8  # Normalize by number of channels
    p2p_csw, _, _ = csr.calculate_p2p(csw)
    rms_csw = csr.calculate_rms(csw)
    snr_csw = p2p_csw / (2 * rms_csw)
    rpr_csw, _, _, _ = csr.calculate_rpr(csw, time)
    impulsivity_csw, cdf_csw = impul.get_impulsivity(csw)
    slope_csw, intercept_csw, _, _ = impul.get_cdf_fit(time, cdf_csw)

    # Store CSW Values
    p2p_csw_values[event] = p2p_csw
    rms_csw_values[event] = rms_csw
    snr_csw_values[event] = snr_csw
    rpr_csw_values[event] = rpr_csw
    impulsivity_csw_values[event] = impulsivity_csw
    slope_csw_values[event] = slope_csw
    intercept_csw_values[event] = intercept_csw

# Function to Plot Scatter
def plot_scatter(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7, color="b")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(filename)
    plt.show()

events = np.arange(num_events)

# Plot Per-Channel Scatter for All Events
for ch in range(num_channels):
    plot_scatter(events, p2p_values[ch, :], "Event Number", f"P2P (Ch {ch})", f"P2P for Channel {ch} Across Events", f"scatter_p2p_ch{ch}.png")
    plot_scatter(events, rms_values[ch, :], "Event Number", f"RMS (Ch {ch})", f"RMS for Channel {ch} Across Events", f"scatter_rms_ch{ch}.png")
    plot_scatter(events, snr_values[ch, :], "Event Number", f"SNR (Ch {ch})", f"SNR for Channel {ch} Across Events", f"scatter_snr_ch{ch}.png")
    plot_scatter(events, rpr_values[ch, :], "Event Number", f"RPR (Ch {ch})", f"RPR for Channel {ch} Across Events", f"scatter_rpr_ch{ch}.png")
    plot_scatter(events, impulsivity_values[ch, :], "Event Number", f"Impulsivity (Ch {ch})", f"Impulsivity for Channel {ch} Across Events", f"scatter_impulsivity_ch{ch}.png")
    plot_scatter(events, slope_values[ch, :], "Event Number", f"Slope (Ch {ch})", f"Slope of CDF Fit for Channel {ch} Across Events", f"scatter_slope_ch{ch}.png")
    plot_scatter(events, intercept_values[ch, :], "Event Number", f"Intercept (Ch {ch})", f"Intercept of CDF Fit for Channel {ch} Across Events", f"scatter_intercept_ch{ch}.png")

# Plot CSW Scatter for All Events
plot_scatter(events, p2p_csw_values, "Event Number", "P2P (CSW)", "P2P for CSW Across Events", "scatter_p2p_csw.png")
plot_scatter(events, rms_csw_values, "Event Number", "RMS (CSW)", "RMS for CSW Across Events", "scatter_rms_csw.png")
plot_scatter(events, snr_csw_values, "Event Number", "SNR (CSW)", "SNR for CSW Across Events", "scatter_snr_csw.png")
plot_scatter(events, rpr_csw_values, "Event Number", "RPR (CSW)", "RPR for CSW Across Events", "scatter_rpr_csw.png")
plot_scatter(events, impulsivity_csw_values, "Event Number", "Impulsivity (CSW)", "Impulsivity for CSW Across Events", "scatter_impulsivity_csw.png")
plot_scatter(events, slope_csw_values, "Event Number", "Slope (CSW)", "Slope of CDF Fit for CSW Across Events", "scatter_slope_csw.png")
plot_scatter(events, intercept_csw_values, "Event Number", "Intercept (CSW)", "Intercept of CDF Fit for CSW Across Events", "scatter_intercept_csw.png")

print("Saved all scatter plots for all events across channels.")

