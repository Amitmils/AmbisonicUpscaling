import spaudiopy as spa
import numpy as np
from py_bank.filterbanks import FilterBank as fb


def create_sh_matrix(N,azi,zen,type='complex'):
    azi = azi.reshape(-1)
    zen = zen.reshape(-1)
    return spa.sph.sh_matrix(N_sph=N,azi=azi,zen=zen,sh_type=type).transpose()


def fft_anm_t(anm_t,fs):
    NFFT = 2 ** np.ceil(np.log2(anm_t.shape[0])).astype(int)  # Equivalent of nextpow2 in MATLAB
    anm_f = np.fft.fft(anm_t, NFFT, axis=0)  # Perform FFT along the rows (axis=0)

    # Remove negative frequencies
    anm_f = anm_f[:NFFT // 2 + 1, :]  # Keep only the positive frequencies

    # Vector of frequencies
    fVec = np.fft.fftfreq(NFFT, 1/fs)  # Create frequency vector
    fVec_pos = fVec[:NFFT // 2 + 1]  # Keep only positive frequencies
    return anm_f,fVec_pos

def divide_anm_t_to_sub_bands(anm_t,fs,num_bins,low_filter_center_freq,DS=2):
    #signal is size [num_samples,(ambi Order+1)^2]
    anm_t = anm_t[::DS]
    fs = fs / DS

    high_filter_center_freq = fs / 2  # centre freq. of highest filter
    num_samples,filter_length = anm_t.shape  # filter bank length
    erb_bank = fb.EqualRectangularBandwidth(filter_length, fs, num_bins, low_filter_center_freq, high_filter_center_freq)
    anm_t_subbands = np.zeros((num_samples,filter_length,num_bins+2)) # num_samples = t | filter_length = num of SH coeff | num_bins + low and high for perfect reconstruction
    for time_sample in range(num_samples):
        erb_bank.generate_subbands(anm_t[time_sample])
        anm_t_subbands[time_sample] = erb_bank.subbands

    #[t,SH_coeff,pass band]
    return anm_t_subbands


def divide_anm_t_to_time_windows(anm_t,window_length):
    #signal is size [num_samples,(ambi Order+1)^2]
    num_samples = anm_t.shape[0]
    anm_t = np.pad(anm_t, ((0, num_samples % window_length), (0, 0)), mode='constant', constant_values=0) 
    windowed_anm_t = np.array_split(anm_t, num_samples // window_length)
    windowed_anm_t = np.stack(windowed_anm_t)
    return windowed_anm_t

