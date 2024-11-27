import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dft(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))

    dft = np.exp(-2j * np.pi * k * n / N)

    return np.dot(dft, signal)

# Fast DFT
def dft_torch(signal):
    signal = torch.tensor(signal, dtype=torch.complex64).to(device)
    return torch.fft.fft(signal)

def frequency_filter(dft_R, frequencies, target_range):
    low, high = target_range
    filtered_signal = np.copy(dft_R)

    for i, freq in enumerate(frequencies):
        if freq < low or freq > high:
            filtered_signal[i] = 0
        
    return filtered_signal

def idft(freq_domain):

    freq_domain = np.asarray(freq_domain, dtype=complex)

    N = len(freq_domain)
    n = np.arange(N)
    k = n.reshape((N, 1))
    idft = np.exp(2j * np.pi * k * n / N)
    return np.dot(idft, freq_domain) / N

# Fast IDFT
def idft_torch(freq_domain):

    freq_domain = torch.tensor(freq_domain).to(device)

    time_domain = torch.fft.ifft(freq_domain)
    return time_domain.cpu().real.numpy()

