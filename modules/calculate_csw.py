import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def align_and_calculate_csw(waveforms):
    """
    Aligns all waveforms relative to Channel 0 using cross-correlation and calculates the Coherently Summed Waveform (CSW).

    Parameters:
        waveforms (2D numpy array): Each row is a waveform from a different channel.

    Returns:
        aligned_waveforms (2D numpy array): Time-aligned waveforms.
        csw (1D numpy array): Coherently Summed Waveform (CSW).
    """
    ref_waveform = waveforms[0]  # Use Channel 0 as reference
    num_channels, num_samples = waveforms.shape
    aligned_waveforms = np.zeros_like(waveforms)
    csw = np.zeros(num_samples)  # Initialize empty CSW

    for i in range(num_channels):
        correlation = np.correlate(waveforms[i], ref_waveform, mode="full")
        shift = np.argmax(correlation) - (num_samples - 1)
        aligned_waveforms[i] = np.roll(waveforms[i], -shift)  # Apply shift
        csw += aligned_waveforms[i]  # Summing coherently

    return aligned_waveforms, csw

def create_csw_gif(aligned_waveforms, time, output_gif="csw_evolution.gif"):
    """
    Creates a GIF showing the gradual formation of the Coherently Summed Waveform (CSW).

    Parameters:
        aligned_waveforms (2D numpy array): Aligned waveforms for summation.
        time (1D numpy array): The time axis corresponding to the waveforms.
        output_gif (str): The filename for the output GIF.

    Returns:
        None (Saves a GIF showing the CSW formation)
    """
    num_channels, num_samples = aligned_waveforms.shape
    csw = np.zeros(num_samples)  # Initialize empty CSW
    frames = []  # Store images for GIF

    # Create temporary directory for frames
    temp_dir = "csw_frames"
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(num_channels):
        csw += aligned_waveforms[i]  # Add next channel to CSW

        # Plot the CSW formation process
        plt.figure(figsize=(10, 5))
        plt.plot(time, csw, color="red", label=f"CSW (Using {i+1} Channels)", linewidth=2)
        
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.title("Formation of Coherently Summed Waveform (CSW)")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)

        # Save frame
        frame_path = f"{temp_dir}/frame_{i+1}.png"
        plt.savefig(frame_path)
        frames.append(imageio.imread(frame_path))
        plt.show()
        plt.close()

    # Create GIF
    imageio.mimsave(output_gif, frames, duration=1, loop=0)

    # Clean up temporary images
    for frame in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, frame))
    os.rmdir(temp_dir)

    print(f"GIF saved as {output_gif}")

# Example Usage
import pandas as pd
file1 = pd.read_csv('noise_traces.csv')
file2 = pd.read_csv('traces.csv')

time = np.array(file1['times'])
waveform = np.array(file2['traces'])

waveforms = np.tile(waveform, (8, 1))  # Simulating 8 channels

# Align and compute CSW
aligned_waveforms, csw = align_and_calculate_csw(waveforms)

# Generate CSW GIF
create_csw_gif(aligned_waveforms, time)

