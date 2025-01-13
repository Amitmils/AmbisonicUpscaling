'''
Created 06/03/2018
@author: Will Wilkinson
'''

import torch

class FilterBank(object):
    """
    Based on Josh McDermott's Matlab filterbank code:
    http://mcdermottlab.mit.edu/Sound_Texture_Synthesis_Toolbox_v1.7.zip

    leny = filter bank length in samples
    fs = sample rate
    N = number of frequency channels / subbands (excluding high-&low-pass which are added for perfect reconstruction)
    low_lim = centre frequency of first (lowest) channel
    high_lim = centre frequency of last (highest) channel
    """
    def __init__(self, leny, fs, N, low_lim, high_lim):
        self.leny = leny
        self.fs = fs
        self.N = N
        self.low_lim = low_lim
        self.high_lim, self.freqs, self.nfreqs = self.check_limits(leny, fs, high_lim)

    def check_limits(self, leny, fs, high_lim):
        if leny % 2 == 0:
            nfreqs = leny // 2
            max_freq = fs // 2
        else:
            nfreqs = (leny - 1) // 2
            max_freq = fs * (leny - 1) // 2 // leny
        freqs = torch.linspace(0, max_freq, nfreqs + 1)
        if high_lim > fs / 2:
            high_lim = max_freq
        return high_lim, freqs, int(nfreqs)

    def generate_subbands(self, signal):
        if signal.shape[0] == 1:  # turn into column vector
            signal = signal.t()
        N = self.filters.shape[1] - 2
        signal_length = signal.shape[0]
        filt_length = self.filters.shape[0]
        
        # watch out: PyTorch fft acts on rows, whereas Matlab fft acts on columns
        fft_sample = torch.fft.fft(signal, dim=0).t()
        
        # generate negative frequencies in right place; filters are column vectors
        if signal_length % 2 == 0:  # even length
            fft_filts = torch.cat([self.filters, torch.flip(self.filters[1:filt_length - 1, :], dims=[0])], dim=0).to(signal.device)
        else:  # odd length
            fft_filts = torch.cat([self.filters, torch.flip(self.filters[1:filt_length, :], dims=[0])], dim=0).to(signal.device)
        
        # multiply by array of column replicas of fft_sample
        tile = fft_sample.reshape(-1,1) @ torch.ones(1, N + 2, device=signal.device, dtype=fft_sample.dtype)
        fft_subbands = fft_filts * tile
        
        # ifft works on rows; imag part is small, probably discretization error?
        self.subbands = torch.real(torch.fft.ifft(fft_subbands.t(), dim=0)).t()


class EqualRectangularBandwidth(FilterBank):
    def __init__(self, leny, fs, N, low_lim, high_lim):
        super().__init__(leny, fs, N, low_lim, high_lim)
        # make cutoffs evenly spaced on an erb scale
        erb_low = self.freq2erb(self.low_lim)
        erb_high = self.freq2erb(self.high_lim)
        erb_lims = torch.linspace(erb_low, erb_high, self.N + 2)
        self.cutoffs = self.erb2freq(erb_lims)
        self.filters = self.make_filters(self.N, self.nfreqs, self.freqs, self.cutoffs)

    def freq2erb(self, freq_Hz):
        n_erb = 9.265 * torch.log(1 + freq_Hz / torch.tensor((24.7 * 9.265)))
        return n_erb

    def erb2freq(self, n_erb):
        freq_Hz = 24.7 * 9.265 * (torch.exp(n_erb / 9.265) - 1)
        return freq_Hz

    def make_filters(self, N, nfreqs, freqs, cutoffs):
        cos_filts = torch.zeros([nfreqs + 1, N], device=freqs.device)
        for k in range(N):
            l_k = cutoffs[k]
            h_k = cutoffs[k + 2]  # adjacent filters overlap by 50%
            l_ind = torch.min(torch.where(freqs > l_k)[0])
            h_ind = torch.max(torch.where(freqs < h_k)[0])
            avg = (self.freq2erb(l_k) + self.freq2erb(h_k)) / 2
            rnge = self.freq2erb(h_k) - self.freq2erb(l_k)
            # map cutoffs to -pi/2, pi/2 interval
            cos_filts[l_ind:h_ind + 1, k] = torch.cos((self.freq2erb(freqs[l_ind:h_ind + 1]) - avg) / rnge * torch.pi)
        # add lowpass and highpass to get perfect reconstruction
        filters = torch.zeros([nfreqs + 1, N + 2], device=freqs.device)
        filters[:, 1:N + 1] = cos_filts
        # lowpass filter goes up to peak of first cos filter
        h_ind = torch.max(torch.where(freqs < cutoffs[1])[0])
        filters[:h_ind + 1, 0] = torch.sqrt(1 - filters[:h_ind + 1, 1] ** 2)
        # highpass filter goes down to peak of last cos filter
        l_ind = torch.min(torch.where(freqs > cutoffs[N])[0])
        filters[l_ind:nfreqs + 1, N + 1] = torch.sqrt(1 - filters[l_ind:nfreqs + 1, N] ** 2)
        return filters

