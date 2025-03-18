import matplotlib.pyplot as plt
import numpy as np
import h5py
import imageio
import os

with h5py.File("event_file.h5", "r") as f:
    waveforms = f["waveform_collect"][:]  # Shape (16, 500, 1230)

# Select the event
event_number = 444
if event_number < 0 or event_number >= 500:
    raise ValueError(f"Invalid event number {event_number}, must be in range 0-499")

waveform_event = waveforms[:, event_number, :]  # Shape (16, 1230)

# Generate a time axis (assuming 0.5 ns per sample)
num_samples = waveform_event.shape[1]
time = np.linspace(0, num_samples * 0.5, num_samples)

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

    for i in [0,1,2,4,5,6]:#range(num_channels):
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
    imageio.mimsave(output_gif, frames, duration=0.0008, loop=0)

    # Clean up temporary images
    for frame in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, frame))
    os.rmdir(temp_dir)

    print(f"GIF saved as {output_gif}")


# Align and compute CSW
aligned_waveforms, csw = align_and_calculate_csw(waveform_event)

# Generate CSW GIF
create_csw_gif(aligned_waveforms, time)

#create_csw_gif(waveform_event[:8], time)



