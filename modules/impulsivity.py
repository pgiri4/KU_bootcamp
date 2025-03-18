import numpy as np
import matplotlib.pyplot as plt

import imageio
import os
import matplotlib.cm as cm
from scipy.stats import linregress

import pandas as pd
from scipy.signal import hilbert 



def get_hilbert_envelope(trace):
    """
    Applies the Hilbert Transform to a given waveform trace,
    then it will give us an envelope trace.
    """
    envelope = np.abs(hilbert(trace))
    return envelope

def get_impulsivity(trace):
    """
    Calculates the impulsivity of a signal (trace).
    """
    envelope = get_hilbert_envelope(trace)
    maxv = np.argmax(envelope)
    envelope_indexes = np.arange(len(envelope))  # indices 0, 1, 2, ...
    closeness = np.abs(envelope_indexes - maxv)  # distance from max value

    sorted_envelope = np.array([x for _, x in sorted(zip(closeness, envelope))])

    cdf = np.cumsum(sorted_envelope**2)
    cdf = cdf / cdf[-1]

    
    impulsivity = (np.mean(cdf) * 2.0) - 1.0
    if impulsivity < 0:
        impulsivity = 0.0

    return impulsivity,cdf


def get_cdf_fit(t,cdf):
    slope, intercept, r_value, p_value, std_err = linregress(t, cdf)
    cdf_fit = slope * t + intercept
    return slope,intercept,t,cdf_fit

