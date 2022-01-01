import numpy as np
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
from scipy.signal import find_peaks
from scipy import signal


def load_mp4(fn):
    vidcap = cv2.VideoCapture(fn)

    frames = []

    skip = 60
    ncols = 13
    nrows = 26
    for n in range(skip + ncols * nrows):
        success, frame = vidcap.read()
        if n >= skip:
            frames.append(frame)

    frames = np.stack(frames)
    return frames


def crop_frames(frames):
    s = frames.shape
    if s[2] > s[1]:
        frames = frames[:,:,(1280//2):,:]
        width = 720
    elif s[1] > s[2]:
        frames = frames[:,0:(1280//2),:,:]
        width = 640
    else:
        print("Square, no cropping")

    return frames


def plot_tiled(frames, ncols, suffix):
    nrows = int(np.floor(frames.shape[0] / ncols))
    frames = frames[1:(nrows * ncols + 1)]
    frames2 = frames.reshape(nrows, ncols, frames.shape[1], frames.shape[2], 3)

    # add border
    frames2[:,:,0,:,:] = 255
    frames2[:,:,:,0,:] = 255

    tiled = np.hstack(np.hstack(frames2))

    cv2.imwrite(f'images/tiled_raw{suffix}.jpg', tiled)


def analyse_mp4(fn, suffix, threshold=100):
    frames = load_mp4(fn)

    frames = crop_frames(frames)

    # average
    frames = frames.astype('float64') - frames.mean(axis=0, keepdims=1)

    m = (frames > threshold).mean(axis=(1,2,3))
    m = m - m.min()

    peaks, _ = find_peaks(m, prominence=m.max()/4, distance=5)
    peak_diff = np.diff(peaks)
    # remove outliers
    peak_diff = peak_diff[peak_diff < np.median(peak_diff) * 1.5]
    peak_diff = peak_diff[peak_diff > np.median(peak_diff) * 0.75]
    period = peak_diff.mean()

    plt.figure()
    plt.plot(m)
    plt.vlines(np.arange(peaks[0], len(m), period), ymin=m.min(), ymax=m.max(), linestyles='dotted')
    plt.title(f'{round(period,2)} frames per cycle')
    plt.xlabel("Frame")
    plt.ylabel("Fraction of pixels above threshold")
    plt.savefig(f"images/mean_brightness{suffix}.png")

    plot_tiled(frames, int(round(period + 0.0)), suffix)


def fft_analysis(scale, start_time, end_time, plot_filename, sampling_rate, line=None, n_labels=5):
    m = scale[int(start_time*sampling_rate):int(end_time*sampling_rate)]
    ft = np.fft.rfft(m)
    freqs = np.fft.rfftfreq(len(m), 1/sampling_rate) # Get frequency axis from the time axis
    peaks, peak_props = find_peaks(abs(ft), prominence=abs(ft).max()/5, distance=220)
    peak_freqs = freqs[peaks[(-peak_props['prominences']).argsort()]][0:n_labels]
    print(peak_freqs)
    print(peak_props['prominences'][(-peak_props['prominences']).argsort()[0:n_labels]]/abs(ft).max())
    peak_mags = abs(ft)[peaks[(-peak_props['prominences']).argsort()]][0:n_labels]
    binwidth = round(np.diff(freqs).mean(), 2)
    plt.figure()
    plt.loglog(freqs, abs(ft))
    plt.xlim(5, 2200)
    plt.yscale('log')
    plt.ylim(0.01, abs(ft).max()*1.2)
    plt.vlines(line, ymin=0.01, ymax=abs(ft).max(), linestyles='dotted')
    if line:
        peak_freqs = np.append(peak_freqs, line)
        peak_mags = np.append(peak_mags, [abs(ft).max() * 0.95]*len(line))
    plt.ylabel("Amplitude")
    plt.xlabel(f"Frequency Bin (Hz), Binwidth: {binwidth}")
    plt.title(f"Fan setting: {plot_filename}")
    for peak_freq, peak_mag in zip(peak_freqs, peak_mags):
        plt.annotate(f'{round(peak_freq,2)}', (peak_freq, peak_mag), (peak_freq * 1.2, peak_mag * 0.9),
            arrowprops={'arrowstyle':ArrowStyle.CurveB()})
    plt.savefig(f"images/FFT_{plot_filename}.png")


def analyse_wav(fn):
    scale, sampling_rate = librosa.load("fan2.wav", sr=None)

    X = librosa.stft(scale, n_fft=2048*32) # win_length = n_fft, win_length // 4 = 2048*32 // 4
    X.shape
    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure()
    librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz', cmap='viridis', hop_length=2048*32 // 4)
    plt.ylim(0,45)
    plt.xlim(0,100)
    plt.ylabel("Frequency (Hz)")
    plt.hlines([240/23.33, 240/7.27, 240/14.45], xmin=0, xmax=150, linestyles=(0, (2, 10)), color='red')
    plt.colorbar(format="%+2.0f dB")
    plt.savefig(f"images/spectrogram.png")


    fft_analysis(scale, 90, 120, "Off", sampling_rate, None, 3)
    fft_analysis(scale, 10, 34, "Low", sampling_rate, [10.29, 10.29*11], 3)
    fft_analysis(scale, 39, 53, "Medium", sampling_rate)
    fft_analysis(scale, 60, 78, "High", sampling_rate, [33.11*11], 2)


def main():
    analyse_wav('data/fan2.wav')

    analyse_mp4('data/PXL_20211205_164203985.mp4', '_slow')

    analyse_mp4('data/PXL_20211205_164344502.mp4', '_medium')

    analyse_mp4('data/PXL_20211205_164438618.mp4', '_high', threshold=70)


#exec(open('tile_mp4.py').read())
if __name__ == "__main__":
    main()
