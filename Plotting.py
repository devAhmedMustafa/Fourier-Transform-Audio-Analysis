import matplotlib.pyplot as plt
import librosa


def plot_signal(signal, sr):

    time = librosa.times_like(signal, sr=sr)

    plt.figure(figsize=(12, 6))
    plt.plot(time, signal, color="blue")
    plt.show()