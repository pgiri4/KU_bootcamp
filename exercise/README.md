Examples

Data Overview

There are two CSV files containing waveform data:

traces.csv - Contains signal-like waveform arrays

noise_traces.csv - Contains noise-like waveform arrays

Running Single Event Analysis

To analyze a single waveform and generate plots, run:

python plotting.py

✅ Expected Outputs:

📸 snr_rpr.png (A plot that shows the following calculations)

How P2P (Peak-to-Peak) is calculated

How waveform is divided into segments for RMS calculation

How RPR (Relative Power Ratio) is estimated

Printout of P2P, RMS, SNR, and RPR values

📈 Other plots include:

CSW (Coherently Summed Waveform)

CDF (Cumulative Distribution Function)

Average CDF

Impulsivity & CDF fit with slope & intercept

🧪 Running Multi-Event Analysis

There is an HDF5 file (event_file.h5) that contains 500 events (both noise & signal-like waveforms).

🔍 Analyze a Single Event

To explore a specific event, use:

python multi_event_example1.py --h5_file event_file.h5 --event_number 325

Choose any event number (0-499) and explore how it looks, along with various plots.

This will generate multiple PNG files related to waveform characteristics.

�� Analyze All Events

To process all 500 events and visualize relationships between different waveform properties, run:

python multi_event_example2.py --h5_file event_file.h5

This will extract all variables and generate scatter plots showing their correlations.

🛠 Installation & Dependencies

Ensure you have the required dependencies installed:

pip install numpy matplotlib pandas scipy h5py

�� How to Use This Repository?

✔️ Run the scripts as described above.✔️ Choose different event numbers in multi_event_example1.py.✔️ Analyze waveform characteristics across multiple events using multi_event_example2.py.



