import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import itertools
import pandas as pd
import sys
import os

# Get the absolute path of the modules directory (one level up + modules)
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules"))

# Add to Python path
sys.path.append(module_path)


import calculate_snr_rpr as csr
import impulsivity as impul





file1 = pd.read_csv('noise_traces.csv')
file2 = pd.read_csv('traces.csv')

t = np.array(file1['times'])
sig_waveform = np.array(file2['traces'])
noise_like_waveform = file1['traces']

V_p2p,best_start_idx, best_end_idx = csr.calculate_p2p(sig_waveform)
rms = csr.calculate_rms(sig_waveform)
SNR = V_p2p/(2*rms)
print('p2p', V_p2p,'rms ', rms,'SNR ', SNR)
rpr, max_rpr_idx, smoothed_waveform, sum_win = csr.calculate_rpr(sig_waveform,t)
print('rpr' ,rpr)

def plot_waveform(time, waveform, p2p, best_start_idx, best_end_idx, rms, rpr, max_rpr_idx, smoothed_waveform, sum_win):
    """
    Plots:
    1. Original waveform with highlighted max P2P region.
    2. Segmented waveform.
    3. Smoothed waveform for RPR calculation (with max power region highlighted).
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # 1. Original waveform with highlighted P2P region
    highlight_indices = list(range(max(0, best_start_idx - 10), min(len(time), best_start_idx + 11))) + \
                        list(range(max(0, best_end_idx - 10), min(len(time), best_end_idx + 11)))
    highlight_indices = sorted(set(highlight_indices))

    for i in range(len(time) - 1):
        if i in highlight_indices:
            axs[0].plot(time[i:i+2], waveform[i:i+2], color='black', linewidth=1.5)
        else:
            axs[0].plot(time[i:i+2], waveform[i:i+2], color='royalblue', linewidth=1)

    if best_start_idx is not None and best_end_idx is not None:
        axs[0].scatter(time[best_start_idx], waveform[best_start_idx], color='green', s=100, edgecolor='black', label="Max P2P Peak", zorder=5)
        axs[0].scatter(time[best_end_idx], waveform[best_end_idx], color='red', s=100, edgecolor='black', label="Min P2P Valley", zorder=5)

    axs[0].set_title(f"Waveform with Maximum Peak-to-Peak (SNR: {p2p:.2f})")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Waveform Segments
    num_segments = 8
    segment_size = len(waveform) // num_segments
    colors = itertools.cycle(plt.cm.get_cmap("tab10").colors)

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(waveform)
        axs[1].plot(time[start_idx:end_idx], waveform[start_idx:end_idx], color=next(colors), linewidth=1.5)

    axs[1].set_title(f"Waveform Segments (RMS: {rms:.2f})")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # 3. Smoothed Waveform with Max Power Region Highlighted (RPR)
    highlight_start = max(0, max_rpr_idx - sum_win // 2)
    highlight_end = min(len(time) - 1, max_rpr_idx + sum_win // 2)

    axs[2].plot(time, waveform, label="Original Waveform", color='blue', alpha=0.8)
    axs[2].plot(time, smoothed_waveform, label="Smoothed Waveform", color='red', linewidth=2)

    # Highlight the 25 ns window with the highest power
    for i in range(highlight_start, highlight_end):
        axs[2].plot(time[i:i+2], smoothed_waveform[i:i+2], color='black', linewidth=2)

    axs[2].set_title(f"Smoothed Waveform (RPR: {rpr:.2f}) with Max Power Region Highlighted")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Amplitude")
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("snr_rpr.png")
    plt.close()

plot_waveform(t, sig_waveform,V_p2p, best_start_idx, best_end_idx, rms, rpr, max_rpr_idx, smoothed_waveform, sum_win)     



impulsivity,cdf = impul.get_impulsivity(sig_waveform)
n_imp,n_cdf = impul.get_impulsivity(noise_like_waveform)

slope,intercept,t,cdf_fit = impul.get_cdf_fit(t,cdf)
slope1,intercept1,t,cdf_noise_fit = impul.get_cdf_fit(t,n_cdf)

plt.figure(figsize=(8, 6))
plt.plot(t,cdf,label = 'Signal cdf',color = 'b')
plt.plot(t,n_cdf,label = 'noise cdf',color = 'orange')
plt.axhline(np.mean(cdf), label = 'average_signal_cdf',linestyle = '--',color = 'blue')
plt.axhline(np.mean(n_cdf), label = 'average_noise_cdf',linestyle = '--',color = 'orange')
plt.legend()

plt.xlabel("Normalized Time")
plt.ylabel("Normalized CDF")
plt.legend()
plt.show()
plt.savefig("av_cdf.png")
plt.close()
print('noise ', slope, intercept)

print('sig ', slope1, intercept1)

plt.plot(t,cdf,label = 'Signal cdf',color = 'b')                                                                                           
plt.plot(t,cdf_fit,label = 'Signal cdf fit',color = 'b',linestyle = '--')                                                              
plt.plot(t,n_cdf,label = 'noise cdf',color = 'orange')                                                                                     
plt.plot(t,cdf_noise_fit,label = 'Signal cdf fit',color = 'orange',linestyle = '--')                                                       
plt.xlabel("Normalized time")                                                                                                              
plt.ylabel("Normalized CDF")                                                                                                               
plt.legend()                                                                                                                               
plt.savefig('cdf_fit.png')        
plt.close()

impulsivity_sig = 2*np.mean(cdf) -1 
impulsivity_noise = 2*np.mean(n_cdf) - 1 


plt.plot(t,cdf,label = 'Signal cdf',color = 'b')                                                                                           
plt.plot(t,n_cdf,label = 'noise cdf',color = 'orange')
#plt.plot(t,cdf_fit,label = 'Signal cdf fit',color = 'b',linestyle = '--')                                                                  
plt.axhline(impulsivity_sig,label = 'impulsivity Signal ',color = 'b',linestyle = '--')                                                                                     
plt.axhline(impulsivity_noise,label = 'impulsivity noise',color = 'orange',linestyle = '--')                                                       
plt.xlabel("Normalized time")                                                                                                              
plt.ylabel("Normalized CDF")                                                                                                               
plt.legend()                                                                                                                               
plt.savefig('impulsivity.png')                                                                                                             
plt.close()      


