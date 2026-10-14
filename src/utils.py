import matplotlib.pyplot as plt
import numpy as np



hz_to_mel = lambda freq: 2595.0 * np.log10(1.0 + (freq / 700.0))
mel_to_hz = lambda mels: 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

def plot_mel_spec(spec, f_min_mel, f_max_mel, n_mels, 
              hop_length, len_waveform, sample_rate,
              title=None, aspect=1):


    # calculate mel freq bins
    m_min = hz_to_mel(f_min_mel)
    m_max = hz_to_mel(f_max_mel)

    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)[1:-1]

    f_display = np.concatenate((np.arange(1, 7)*1000, np.array([10000])))
    mel_display = np.interp(f_display, f_pts, np.arange(n_mels))

    x_labels = np.arange(np.round(len_waveform / sample_rate))
    x_ticks = x_labels * sample_rate / hop_length

    plt.matshow(spec, aspect=aspect, origin="lower")
    plt.gca().xaxis.tick_bottom() 
    plt.xlabel("Temps (s)")
    plt.ylabel("Fréquence (Hz)")
    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.yticks(ticks=mel_display, labels=f_display)

    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()