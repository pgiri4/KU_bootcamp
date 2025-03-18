import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.ndimage import uniform_filter1d
import itertools
import pandas as pd

# Load waveform data
file1 = pd.read_csv('noise_traces.csv')
file2 = pd.read_csv('traces.csv')

time = np.array(file1['times'])
waveform = np.array(file2['traces'])
#waveform = np.array(file1['traces'])
def calculate_p2p(waveform):
    """
    Identifies and calculates the peak-to-peak (P2P) value using a sliding window approach.
    """
    window_size = 20  # 20-sample window
    max_p2p = 0
    best_start_idx = None
    best_end_idx = None

    for i in range(len(waveform) - window_size + 1):
        window = waveform[i : i + window_size]
        p2p_value = np.max(window) - np.min(window)

        if p2p_value > max_p2p:
            max_p2p = p2p_value
            best_start_idx = i + np.argmax(window)
            best_end_idx = i + np.argmin(window)

    return max_p2p, best_start_idx, best_end_idx

def calculate_rms(waveform, num_segments=8):
    """
    Divides the waveform into 8 segments and returns the mean of the two lowest RMS values.
    """
    segment_size = len(waveform) // num_segments
    rms_values = []

    for i in range(num_segments):
        segment = waveform[i * segment_size : (i + 1) * segment_size]
        rms_values.append(np.sqrt(np.mean(segment**2)))

    lowest_rms_mean = np.mean(sorted(rms_values)[:2])  # Mean of two lowest RMS values
    return lowest_rms_mean

def calculate_rpr(waveform, time):
    """
    Computes the Root Power Ratio (RPR) by finding the 25 ns window with max power.
    Returns the RPR value, the index of the highest power window, and the smoothed waveform.
    """
    dt = time[1] - time[0]  # Time resolution
    sum_win = int(np.round(25 / dt))  # Convert 25 ns to number of samples

    # Compute power in sliding window
    smoothed_waveform = np.sqrt(uniform_filter1d(waveform**2, size=sum_win, mode='constant'))
    max_rpr_idx = np.argmax(smoothed_waveform)  # Find the window with max power
    rpr = np.max(smoothed_waveform) / calculate_rms(smoothed_waveform)

    return rpr, max_rpr_idx, smoothed_waveform, sum_win

