import numpy as np
import pylab
import skimage
import wave

from speech_emotion_recognition.iemocap_utils.features import short_term_feature_extraction


def calculate_acoustic_features(frames, freq, options):
    # double the window duration
    window_sec = 0.08
    window_n = int(freq * window_sec)

    st_f = short_term_feature_extraction(frames, freq, window_n, window_n / 2)

    if st_f.shape[1] > 2:
        i0 = 1
        i1 = st_f.shape[1] - 1
        if i1 - i0 < 1:
            i1 = i0 + 1

        deriv_st_f = np.zeros((st_f.shape[0], i1 - i0), dtype=float)
        for i in range(i0, i1):
            i_left = i - 1
            i_right = i + 1
            deriv_st_f[:st_f.shape[0], i - i0] = st_f[:, i]
        return deriv_st_f
    elif st_f.shape[1] == 2:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f
    else:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f


def pad_sequence_into_array(Xs, maxlen=None):
    Nsamples = len(Xs)
    Xout = np.ones(shape=[Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(0.0, dtype=Xs[0].dtype)
    Mask = np.zeros(shape=[Nsamples, maxlen], dtype=Xout.dtype)
    for i in range(Nsamples):
        x = Xs[i]
        trunc = x[:maxlen]
        Xout[i, :len(trunc)] = trunc
        Mask[i, :len(trunc)] = 1
    return Xout, Mask


def generate_spectrogram(wav_file, view=False):
    MAX_SPETROGRAM_LENGTH = 999  # 8 sec
    MAX_SPETROGRAM_TIME_LENGTH_POOLED = 128
    MAX_SPETROGRAM_FREQ_LENGTH_POOLED = 128

    def get_wav_info(wav_file):
        wav = wave.open(wav_file, 'r')
        frames = wav.readframes(-1)
        sound_info = pylab.fromstring(frames, 'Int16')
        frame_rate = wav.getframerate()
        wav.close()
        return sound_info, frame_rate

    """Based on https://dzone.com/articles/generating-audio-spectrograms"""

    """Loading wav file"""
    sound_info, frame_rate = get_wav_info(wav_file)

    """Creating spectrogram"""
    spec, freqs, times, axes = pylab.specgram(sound_info, Fs=frame_rate)

    """Checking dimensions of spectrogram"""
    assert spec.shape[0] == freqs.shape[0] and spec.shape[1] == times.shape[0], "Original dimensions of spectrogram are inconsistent"

    """Extracting a const length spectrogram"""
    times = times[:MAX_SPETROGRAM_LENGTH]
    spec = spec[:, :MAX_SPETROGRAM_LENGTH]
    assert spec.shape[1] == times.shape[0], "Dimensions of spectrogram are inconsistent after change"

    spec_log = np.log(spec)
    spec_pooled = skimage.measure.block_reduce(spec_log, (1, 8), np.mean)
    spec_cropped = spec_pooled[:MAX_SPETROGRAM_FREQ_LENGTH_POOLED, :MAX_SPETROGRAM_TIME_LENGTH_POOLED]
    spectrogram = np.zeros((MAX_SPETROGRAM_FREQ_LENGTH_POOLED, MAX_SPETROGRAM_TIME_LENGTH_POOLED))
    spectrogram[:, :spec_cropped.shape[1]] = spec_cropped

    if view:
        plt.imshow(spec_cropped, cmap='hot', interpolation='nearest')
        plt.show()

    return spectrogram