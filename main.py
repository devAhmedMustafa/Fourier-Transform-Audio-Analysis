from Plotting import *
from DFT import *
import librosa
import soundfile as sf
import json

def decompose_audio(audio_file, freqRanges):

    y, sr = librosa.load(f"assets/{audio_file}", sr=None)

    N = len(y)

    frequencies = np.fft.fftfreq(N, d=1/sr)
    dft_r = None

    dft_r = dft_torch(y)

    filtereds = [
        frequency_filter(dft_r, frequencies, freqRange) for freqRange in freqRanges
    ]

    f_signals = [
        idft_torch(filtered) for filtered in filtereds
    ]

    f_signals = [
        np.int16(f_signal / np.max(np.abs(f_signal)) * 32767) for f_signal in f_signals
    ]

    for i in range(len(f_signals)):

        sf.write(f"filtered/{audio_file.replace('.', '')}-filtered_{freqRanges[i]}.wav", f_signals[i], sr)

        plot_signal(f_signals[i], sr)

def compose_audio(audio_files, output_naming=""):
    
    Y = [
        librosa.load(file, sr=None)[0] for file in audio_files
    ]

    SR = [
        librosa.load(file, sr=None)[1] for file in audio_files
    ]

    # Get min Length
    min_len = min(len(y) for y in Y)

    # Ensure every file has the same length
    for i in range(len(Y)):
        Y[i] = Y[i][:min_len]

    FFTs = [
        dft_torch(y) for y in Y
    ]

    # Combine FFTs
    combined = torch.sum(torch.stack(FFTs), dim=0)

    # Compute IDFTs of combined
    combined_idft = idft_torch(combined)
    combined_idft = np.int16(combined_idft / np.max(np.abs(combined_idft)) * 32767)

    sf.write(f"composed/composed_{output_naming}.wav", combined_idft, SR[0])
    print(f"Composed audio saved")


if __name__ == "__main__":

    freqRanges = [
        (0, 100),
        (100, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
        (8000, 16000),
    ]

    audio_primary_name = 'milk-shake'

    decompose_audio(f'{audio_primary_name}.mp3', freqRanges)

    audio_files = [
        f'filtered/{audio_primary_name}mp3-filtered_{fr}.wav' for fr in freqRanges
    ]

    compose_audio(audio_files, audio_primary_name)
