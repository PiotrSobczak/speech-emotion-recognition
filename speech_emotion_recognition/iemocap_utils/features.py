import sys
import time
import os
import glob
import numpy
import math
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import lfilter, hamming

eps = 0.00000001


""" Time-domain audio features """


def short_term_ZCR(frame):
    """Calculating zero crossing rate of frame"""
    count = len(frame)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return (numpy.float64(countZ) / numpy.float64(count-1.0))


def short_term_energy(frame):
    """Calculating signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))


def short_term_energy_entropy(frame, numOfShortBlocks=10):
    """Calculating entropy of energy"""
    Eol = numpy.sum(frame ** 2)    # total frame energy
    L = len(frame)
    subWinLength = int(numpy.floor(L / numOfShortBlocks))
    if L != subWinLength * numOfShortBlocks:
            frame = frame[0:subWinLength * numOfShortBlocks]
    # subWindows is of size [numOfShortBlocks x L]
    subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -numpy.sum(s * numpy.log2(s + eps))
    return Entropy


""" Frequency-domain audio features """


def short_term_spectral_centroid_and_spread(X, fs):
    """Calculating spectral centroid of frame (given abs(FFT))"""
    ind = (numpy.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = numpy.sum(ind * Xt)
    DEN = numpy.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = numpy.sqrt(numpy.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)


def short_term_spectral_entropy(X, numOfShortBlocks=10):
    """Calculating the spectral entropy"""
    L = len(X)                         # number of frame samples
    Eol = numpy.sum(X ** 2)            # total spectral energy

    subWinLength = int(numpy.floor(L / numOfShortBlocks))   # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]

    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)                      # compute spectral sub-energies
    En = -numpy.sum(s*numpy.log2(s + eps))                                    # compute spectral entropy

    return En


def short_term_spectral_flux(X, Xprev):
    """
    Calculating the spectral flux feature of the current frame
    ARGUMENTS:
        X:        the abs(fft) of the current frame
        Xpre:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = numpy.sum(X + eps)
    sumPrevX = numpy.sum(Xprev + eps)
    F = numpy.sum((X / sumX - Xprev/sumPrevX) ** 2)

    return F


def short_term_spectral_rolloff(X, c, fs):
    """Calculating spectral roll-off"""
    totalEnergy = numpy.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    CumSum = numpy.cumsum(X ** 2) + eps
    [a, ] = numpy.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = numpy.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)


def short_term_harmonic(frame, fs):
    """
    Calculating harmonic ratio and pitch
    """
    M = numpy.round(0.016 * fs) - 1
    R = numpy.correlate(frame, frame, mode='full')

    g = R[len(frame)-1]
    R = R[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = numpy.nonzero(numpy.diff(numpy.sign(R)))

    if len(a) == 0:
        m0 = len(R)-1
    else:
        m0 = a[0]
    if M > len(R):
        M = len(R) - 1

    Gamma = numpy.zeros((M), dtype=numpy.float64)
    CSum = numpy.cumsum(frame ** 2)
    Gamma[m0:M] = R[m0:M] / (numpy.sqrt((g * CSum[M:m0:-1])) + eps)

    ZCR = short_term_ZCR(Gamma)

    if ZCR > 0.15:
        HR = 0.0
        f0 = 0.0
    else:
        if len(Gamma) == 0:
            HR = 1.0
            blag = 0.0
            Gamma = numpy.zeros((M), dtype=numpy.float64)
        else:
            HR = numpy.max(Gamma)
            blag = numpy.argmax(Gamma)

        # Get fundamental frequency:
        f0 = fs / (blag + eps)
        if f0 > 5000:
            f0 = 0.0
        if HR < 0.1:
            f0 = 0.0

    return (HR, f0)


def mfcc_init_filter_banks(fs, nfft):
    """
    Calculating the triangular filterbank for MFCC computation (used in the short_term_feature_extraction function before the short_term_MFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    # filter bank params:
    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt
    #print (str(nFiltTotal))
    #print (str(nfft))

    # Compute frequency points of the triangle:
    freqs = numpy.zeros(nFiltTotal+2)
    freqs[:numLinFiltTotal] = lowfreq + numpy.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** numpy.arange(1, numLogFilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    
    
    fbank = numpy.zeros((int(nFiltTotal), int(nfft)))
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i+1]
        highTrFreq = freqs[i+2]

        lid = numpy.arange(numpy.floor(lowTrFreq * nfft / fs) + 1, numpy.floor(cenTrFreq * nfft / fs) + 1, dtype=numpy.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = numpy.arange(numpy.floor(cenTrFreq * nfft / fs) + 1, numpy.floor(highTrFreq * nfft / fs) + 1, dtype=numpy.int)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs


def short_term_MFCC(X, fbank, nceps):
    """
    Calculating the MFCCs of a frame, given the fft mag
    ARGUMENTS:
        X:        fft magnitude abs(FFT)
        fbank:    filter bank (see mfcc_init_filter_banks)
    RETURN
        ceps:     MFCCs (13 element vector)
    Note:    MFCC calculation is, in general, taken from the scikits.talkbox library (MIT Licence),
    #    with a small number of modifications to make it more compact and suitable for the pyAudioAnalysis Lib
    """

    mspec = numpy.log10(numpy.dot(X, fbank.T)+eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps


def short_term_chroma_features_init(nfft, fs):
    """
    This function initializes the chroma matrices used in the calculation of the chroma features
    """
    freqs = numpy.array([((f + 1) * fs) / (2 * int(nfft)) for f in range(int(nfft))])
    Cp = 27.50

    nChroma = numpy.round(12.0 * numpy.log2(freqs / Cp)).astype(int)

    nFreqsPerChroma = numpy.zeros((nChroma.shape[0], ))

    uChroma = numpy.unique(nChroma)
    for u in uChroma:
        idx = numpy.nonzero(nChroma == u)
        nFreqsPerChroma[idx] = idx[0].shape
    return nChroma, nFreqsPerChroma


def short_term_chroma_features(X, fs, nChroma, nFreqsPerChroma):
    chromaNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = X**2
    C = numpy.zeros((nChroma.shape[0],))
    C[nChroma] = spec
    C /= nFreqsPerChroma[nChroma]
    finalC = numpy.zeros((12, 1))
    newD = int(numpy.ceil(C.shape[0] // 12.0) * 12)
    C2 = numpy.zeros((newD, ))
    #C2[0:C.shape[0]] = C
    C2 = C2.reshape(int(numpy.ceil(C2.shape[0]//12)), 12)
    #for i in range(12):
    #    finalC[i] = numpy.sum(C[i:C.shape[0]:12])
    finalC = numpy.matrix(numpy.sum(C2, axis=0)).T
    finalC /= spec.sum()

    return chromaNames, finalC


def short_term_chromagram(signal, Fs, Win, Step, PLOT=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a numpy array (nFFT x numOfShortTermWindows)
    ARGUMENTS:
        signal:      the input signal samples
        Fs:          the sampling freq (in Hz)
        Win:         the short-term window size (in samples)
        Step:        the short-term window step (in samples)
        PLOT:        flag, 1 if results are to be ploted
    RETURNS:
    """
    Win = int(Win)
    Step = int(Step)
    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX - DC)

    N = len(signal)        # total number of signals
    curPos = 0
    countFrames = 0
    nfft = int(Win / 2)
    nChroma, nFreqsPerChroma = short_term_chroma_features_init(nfft, Fs)
    chromaGram = numpy.array([], dtype=numpy.float64)

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos:curPos + Win]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)
        chromaNames, C = short_term_chroma_features(X, Fs, nChroma, nFreqsPerChroma)
        C = C[:, 0]
        if countFrames == 1:
            chromaGram = C.T
        else:
            chromaGram = numpy.vstack((chromaGram, C.T))
    FreqAxis = chromaNames
    TimeAxis = [(t * Step) / Fs for t in range(chromaGram.shape[0])]

    if (PLOT):
        fig, ax = plt.subplots()
        chromaGramToPlot = chromaGram.transpose()[::-1, :]
        Ratio = chromaGramToPlot.shape[1] / (3*chromaGramToPlot.shape[0])
        chromaGramToPlot = numpy.repeat(chromaGramToPlot, Ratio, axis=0)
        imgplot = plt.imshow(chromaGramToPlot)
        Fstep = int(nfft / 5.0)
        ax.set_yticks(range(Ratio / 2, len(FreqAxis) * Ratio, Ratio))
        ax.set_yticklabels(FreqAxis[::-1])
        TStep = countFrames / 3
        TimeTicks = range(0, countFrames, TStep)
        TimeTicksLabels = ['%.2f' % (float(t * Step) / Fs) for t in TimeTicks]
        ax.set_xticks(TimeTicks)
        ax.set_xticklabels(TimeTicksLabels)
        ax.set_xlabel('time (secs)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return (chromaGram, TimeAxis, FreqAxis)


def phormants(x, Fs):
    N = len(x)
    w = numpy.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w   
    x1 = lfilter([1], [1., 0.63], x1)
    
    # Get LPC.    
    ncoeff = 2 + Fs / 1000
    A, e, k = lpc(x1, ncoeff)    
    #A, e, k = lpc(x1, 8)

    # Get roots.
    rts = numpy.roots(A)
    rts = [r for r in rts if numpy.imag(r) >= 0]

    # Get angles.
    angz = numpy.arctan2(numpy.imag(rts), numpy.real(rts))

    # Get frequencies.    
    frqs = sorted(angz * (Fs / (2 * math.pi)))

    return frqs


def beat_extraction(st_features, winSize, PLOT=False):
    """
    This function extracts an estimate of the beat rate for a musical signal.
    ARGUMENTS:
     - st_features:     a numpy array (numOf_features x numOfShortTermWindows)
     - winSize:        window size in seconds
    RETURNS:
     - BPM:            estimates of beats per minute
     - Ratio:          a confidence measure
    """

    # _features that are related to the beat tracking task:
    toWatch = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    maxBeatTime = int(round(2.0 / winSize))
    HistAll = numpy.zeros((maxBeatTime,))
    for ii, i in enumerate(toWatch):                                        # for each feature
        DifThres = 2.0 * (numpy.abs(st_features[i, 0:-1] - st_features[i, 1::])).mean()    # dif threshold (3 x Mean of Difs)
        [pos1, _] = utilities.peakdet(st_features[i, :], DifThres)           # detect local maxima
        posDifs = []                                                        # compute histograms of local maxima changes
        for j in range(len(pos1)-1):
            posDifs.append(pos1[j+1]-pos1[j])
        [HistTimes, HistEdges] = numpy.histogram(posDifs, numpy.arange(0.5, maxBeatTime + 1.5))
        Hishort_term_centers = (HistEdges[0:-1] + HistEdges[1::]) / 2.0
        HistTimes = HistTimes.astype(float) / st_features.shape[1]
        HistAll += HistTimes
        if PLOT:
            plt.subplot(9, 2, ii + 1)
            plt.plot(st_features[i, :], 'k')
            for k in pos1:
                plt.plot(k, st_features[i, k], 'k*')
            f1 = plt.gca()
            f1.axes.get_xaxis().set_ticks([])
            f1.axes.get_yaxis().set_ticks([])

    if PLOT:
        plt.show(block=False)
        plt.figure()

    # Get beat as the argmax of the agregated histogram:
    I = numpy.argmax(HistAll)
    BPMs = 60 / (Hishort_term_centers * winSize)
    BPM = BPMs[I]
    # ... and the beat ratio:
    Ratio = HistAll[I] / HistAll.sum()

    if PLOT:
        # filter out >500 beats from plotting:
        HistAll = HistAll[BPMs < 500]
        BPMs = BPMs[BPMs < 500]

        plt.plot(BPMs, HistAll, 'k')
        plt.xlabel('Beats per minute')
        plt.ylabel('Freq Count')
        plt.show(block=True)

    return BPM, Ratio


def short_term_spectogram(signal, Fs, Win, Step, PLOT=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a numpy array (nFFT x numOfShortTermWindows)
    ARGUMENTS:
        signal:      the input signal samples
        Fs:          the sampling freq (in Hz)
        Win:         the short-term window size (in samples)
        Step:        the short-term window step (in samples)
        PLOT:        flag, 1 if results are to be ploted
    RETURNS:
    """
    Win = int(Win)
    Step = int(Step)
    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX - DC)

    N = len(signal)        # total number of signals
    curPos = 0
    countFrames = 0
    nfft = int(Win / 2)
    specgram = numpy.array([], dtype=numpy.float64)

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos:curPos+Win]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)

        if countFrames == 1:
            specgram = X ** 2
        else:
            specgram = numpy.vstack((specgram, X))

    FreqAxis = [((f + 1) * Fs) / (2 * nfft) for f in range(specgram.shape[1])]
    TimeAxis = [(t * Step) / Fs for t in range(specgram.shape[0])]

    if (PLOT):
        fig, ax = plt.subplots()
        imgplot = plt.imshow(specgram.transpose()[::-1, :])
        Fstep = int(nfft / 5.0)
        FreqTicks = range(0, int(nfft) + Fstep, Fstep)
        FreqTicksLabels = [str(Fs / 2 - int((f * Fs) / (2 * nfft))) for f in FreqTicks]
        ax.set_yticks(FreqTicks)
        ax.set_yticklabels(FreqTicksLabels)
        TStep = countFrames/3
        TimeTicks = range(0, countFrames, TStep)
        TimeTicksLabels = ['%.2f' % (float(t * Step) / Fs) for t in TimeTicks]
        ax.set_xticks(TimeTicks)
        ax.set_xticklabels(TimeTicksLabels)
        ax.set_xlabel('time (secs)')
        ax.set_ylabel('freq (Hz)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return (specgram, TimeAxis, FreqAxis)


""" Windowing and feature extraction """


def short_term_feature_extraction(signal, Fs, Win, Step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.
    ARGUMENTS
        signal:       the input signal samples
        Fs:           the sampling freq (in Hz)
        Win:          the short-term window size (in samples)
        Step:         the short-term window step (in samples)
    RETURNS
        st_features:   a numpy array (numOf_features x numOfShortTermWindows)
    """

    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / MAX

    N = len(signal)                                # total number of samples
    curPos = 0
    countFrames = 0
    nFFT = Win / 2

    [fbank, freqs] = mfcc_init_filter_banks(Fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation
    nChroma, nFreqsPerChroma = short_term_chroma_features_init(nFFT, Fs)


    numOfPitches = 5
    numOfPeaks = 10
    numOfTimeSpectral_features = 8
    numOfHarmonic_features = 0
    nceps = 13
    numOfChroma_features = 13

    totalNumOf_features = numOfTimeSpectral_features + nceps + numOfHarmonic_features + numOfChroma_features
#    totalNumOf_features = numOfTimeSpectral_features + nceps + numOfHarmonic_features
    st_features = numpy.array([], dtype=numpy.float64)

    while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
        countFrames += 1
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:int(nFFT)]                                    # normalize fft
        X = X / len(X)
        if countFrames == 1:
            Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)
        curFV = numpy.zeros((totalNumOf_features, 1))
        curFV[0] = short_term_ZCR(x)                              # zero crossing rate
        curFV[1] = short_term_energy(x)                           # short-term energy
        curFV[2] = short_term_energy_entropy(x)                    # short-term entropy of energy
        [curFV[3], curFV[4]] = short_term_spectral_centroid_and_spread(X, Fs)    # spectral centroid and spread
        curFV[5] = short_term_spectral_entropy(X)                  # spectral entropy
        curFV[6] = short_term_spectral_flux(X, Xprev)              # spectral flux
        curFV[7] = short_term_spectral_rolloff(X, 0.90, Fs)        # spectral rolloff
        curFV[numOfTimeSpectral_features:numOfTimeSpectral_features+nceps, 0] = short_term_MFCC(X, fbank, nceps).copy()    # MFCCs

        chromaNames, chromaF = short_term_chroma_features(X, Fs, nChroma, nFreqsPerChroma)
        curFV[numOfTimeSpectral_features + nceps: numOfTimeSpectral_features + nceps + numOfChroma_features - 1] = chromaF
        numOfCF_features = numOfTimeSpectral_features + nceps + numOfChroma_features

        curFV[numOfCF_features-1] = chromaF.std()
        
        if countFrames == 1:
            st_features = curFV                                        # initialize feature matrix (if first frame)
        else:
            st_features = numpy.concatenate((st_features, curFV), 1)    # update feature matrix
        Xprev = X.copy()

    return numpy.array(st_features)


def mid_term_feature_extraction(signal, Fs, mtWin, mtStep, stWin, short_term_step):
    """
    Mid-term feature extraction
    """

    mtWinRatio = int(round(mtWin / short_term_step))
    mtStepRatio = int(round(mtStep / short_term_step))

    mid_term_features = []

    st_features = short_term_feature_extraction(signal, Fs, stWin, short_term_step)
    numOf_features = len(st_features)
    numOfStatistics = 2

    mid_term_features = []
    #for i in range(numOfStatistics * numOf_features + 1):
    for i in range(numOfStatistics * numOf_features):
        mid_term_features.append([])

    for i in range(numOf_features):        # for each of the short-term features:
        curPos = 0
        N = len(st_features[i])
        while (curPos < N):
            N1 = curPos
            N2 = curPos + mtWinRatio
            if N2 > N:
                N2 = N
            curSt_features = st_features[i][N1:N2]

            mid_term_features[i].append(numpy.mean(curSt_features))
            mid_term_features[i+numOf_features].append(numpy.std(curSt_features))
            #mid_term_features[i+2*numOf_features].append(numpy.std(curSt_features) / (numpy.mean(curSt_features)+0.00000010))
            curPos += mtStepRatio

    return numpy.array(mid_term_features), st_features


# TODO
def short_term_feature_speed(signal, Fs, Win, Step):

    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / MAX
    # print (numpy.abs(signal)).max()

    N = len(signal)        # total number of signals
    curPos = 0
    countFrames = 0

    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    nlinfil = 13
    nlogfil = 27
    nceps = 13
    nfil = nlinfil + nlogfil
    nfft = Win / 2
    if Fs < 8000:
        nlogfil = 5
        nfil = nlinfil + nlogfil
        nfft = Win / 2

    # compute filter banks for mfcc:
    [fbank, freqs] = mfcc_init_filter_banks(Fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)

    numOfTimeSpectral_features = 8
    numOfHarmonic_features = 1
    totalNumOf_features = numOfTimeSpectral_features + nceps + numOfHarmonic_features
    #st_features = numpy.array([], dtype=numpy.float64)
    st_features = []

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos:curPos + Win]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)
        Ex = 0.0
        El = 0.0
        X[0:4] = 0
        st_features.append(short_term_harmonic(x, Fs))

    return numpy.array(st_features)


""" Feature _extraction Wrappers
 - The first two feature extraction wrappers are used to extract long-term averaged
   audio features for a list of WAV files stored in a given category.
   It is important to note that, one single feature is extracted per WAV file (not the whole sequence of feature vectors)
 """


def dir_wav_feature_extraction(dirName, mtWin, mtStep, stWin, short_term_step, computeBEAT=False):
    """
    This function extracts the mid-term features of the WAVE files of a particular folder.
    The resulting feature vector is extracted by long-term averaging the mid-term features.
    Therefore ONE FEATURE VECTOR is extracted for each WAV file.
    ARGUMENTS:
        - dirName:        the path of the WAVE directory
        - mtWin, mtStep:    mid-term window and step (in seconds)
        - stWin, short_term_step:    short-term window and step (in seconds)
    """

    allmid_term_features = numpy.array([])
    processingTimes = []

    types = ('*.wav', '*.aif',  '*.aiff')
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(dirName, files)))

    wavFilesList = sorted(wavFilesList)

    for wavFile in wavFilesList:
        [Fs, x] = audioBasicIO.readAudioFile(wavFile)            # read file
        t1 = time.clock()
        x = audioBasicIO.stereo2mono(x)                          # convert stereo to mono
        if computeBEAT:                                          # mid-term feature extraction for current file
            [MidTerm_features, st_features] = mid_term_feature_extraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * short_term_step))
            [beat, beatConf] = beat_extraction(st_features, short_term_step)
        else:
            [MidTerm_features, _] = mid_term_feature_extraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * short_term_step))

        MidTerm_features = numpy.transpose(MidTerm_features)
        MidTerm_features = MidTerm_features.mean(axis=0)         # long term averaging of mid-term statistics
        if computeBEAT:
            MidTerm_features = numpy.append(MidTerm_features, beat)
            MidTerm_features = numpy.append(MidTerm_features, beatConf)
        if len(allmid_term_features) == 0:                              # append feature vector
            allmid_term_features = MidTerm_features
        else:
            allmid_term_features = numpy.vstack((allmid_term_features, MidTerm_features))
        t2 = time.clock()
        duration = float(len(x)) / Fs
        processingTimes.append((t2 - t1) / duration)
    if len(processingTimes) > 0:
        print("Feature extraction complexity ratio: {0:.1f} x realtime".format((1.0 / numpy.mean(numpy.array(processingTimes)))))
    return (allmid_term_features, wavFilesList)


def dirs_wav_feature_extraction(dirNames, mtWin, mtStep, stWin, short_term_step, computeBEAT=False):
    '''
    Same as dir_wav_feature_extraction, but instead of a single dir it takes a list of paths as input and returns a list of feature matrices.
    EXAMPLE:
    [features, classNames] =
           a.dirs_wav_feature_extraction(['audioData/classSegmentsRec/noise','audioData/classSegmentsRec/speech',
                                       'audioData/classSegmentsRec/brush-teeth','audioData/classSegmentsRec/shower'], 1, 1, 0.02, 0.02);
    It can be used during the training process of a classification model ,
    in order to get feature matrices from various audio classes (each stored in a seperate path)
    '''

    # feature extraction for each class:
    features = []
    classNames = []
    fileNames = []
    for i, d in enumerate(dirNames):
        [f, fn] = dir_wav_feature_extraction(d, mtWin, mtStep, stWin, short_term_step, computeBEAT=computeBEAT)
        if f.shape[0] > 0:       # if at least one audio file has been found in the provided folder:
            features.append(f)
            fileNames.append(fn)
            if d[-1] == "/":
                classNames.append(d.split(os.sep)[-2])
            else:
                classNames.append(d.split(os.sep)[-1])
    return features, classNames, fileNames


def dir_wav_feature_extraction_no_averaging(dirName, mtWin, mtStep, stWin, short_term_step):
    """
    This function extracts the mid-term features of the WAVE files of a particular folder without averaging each file.
    ARGUMENTS:
        - dirName:          the path of the WAVE directory
        - mtWin, mtStep:    mid-term window and step (in seconds)
        - stWin, short_term_step:    short-term window and step (in seconds)
    RETURNS:
        - X:                A feature matrix
        - Y:                A matrix of file labels
        - filenames:
    """

    allmid_term_features = numpy.array([])
    signalIndices = numpy.array([])
    processingTimes = []

    types = ('*.wav', '*.aif',  '*.aiff')
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(dirName, files)))

    wavFilesList = sorted(wavFilesList)

    for i, wavFile in enumerate(wavFilesList):
        [Fs, x] = audioBasicIO.readAudioFile(wavFile)            # read file
        x = audioBasicIO.stereo2mono(x)                          # convert stereo to mono
        [MidTerm_features, _] = mid_term_feature_extraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * short_term_step))  # mid-term feature

        MidTerm_features = numpy.transpose(MidTerm_features)
#        MidTerm_features = MidTerm_features.mean(axis=0)        # long term averaging of mid-term statistics
        if len(allmid_term_features) == 0:                # append feature vector
            allmid_term_features = MidTerm_features
            signalIndices = numpy.zeros((MidTerm_features.shape[0], ))
        else:
            allmid_term_features = numpy.vstack((allmid_term_features, MidTerm_features))
            signalIndices = numpy.append(signalIndices, i * numpy.ones((MidTerm_features.shape[0], )))

    return (allmid_term_features, signalIndices, wavFilesList)


# The following two feature extraction wrappers extract features for given audio files, however
# NO LONG-TERM AVERAGING is performed. Therefore, the output for each audio file is NOT A SINGLE FEATURE VECTOR
# but a whole feature matrix.
#
# Also, another difference between the following two wrappers and the previous is that they NO LONG-TERM AVERAGING IS PERFORMED.
# In other words, the WAV files in these functions are not used as uniform samples that need to be averaged but as sequences

def mid_term_feature_extractionToFile(fileName, midTermSize, midTermStep, shortTermSize, shortTermStep, outPutFile,
                              storeSt_features=False, storeToCSV=False, PLOT=False):
    """
    This function is used as a wrapper to:
    a) read the content of a WAV file
    b) perform mid-term feature extraction on that signal
    c) write the mid-term feature sequences to a numpy file
    """
    [Fs, x] = audioBasicIO.readAudioFile(fileName)            # read the wav file
    x = audioBasicIO.stereo2mono(x)                           # convert to MONO if required
    if storeSt_features:
        [mid_term_f, short_term_f] = mid_term_feature_extraction(x, Fs, round(Fs * midTermSize), round(Fs * midTermStep), round(Fs * shortTermSize), round(Fs * shortTermStep))
    else:
        [mid_term_f, _] = mid_term_feature_extraction(x, Fs, round(Fs*midTermSize), round(Fs * midTermStep), round(Fs * shortTermSize), round(Fs * shortTermStep))

    numpy.save(outPutFile, mid_term_f)                              # save mt features to numpy file
    if PLOT:
        print("Mid-term numpy file: " + outPutFile + ".npy saved")
    if storeToCSV:
        numpy.savetxt(outPutFile+".csv", mid_term_f.T, delimiter=",")
        if PLOT:
            print("Mid-term CSV file: " + outPutFile + ".csv saved")

    if storeSt_features:
        numpy.save(outPutFile+"_st", short_term_f)                    # save st features to numpy file
        if PLOT:
            print("Short-term numpy file: " + outPutFile + "_st.npy saved")
        if storeToCSV:
            numpy.savetxt(outPutFile+"_st.csv", short_term_f.T, delimiter=",")    # store st features to CSV file
            if PLOT:
                print("Short-term CSV file: " + outPutFile + "_st.csv saved")


def mid_term_feature_extractionToFileDir(dirName, midTermSize, midTermStep, shortTermSize, shortTermStep, storeSt_features=False, storeToCSV=False, PLOT=False):
    types = (dirName + os.sep + '*.wav', )
    filesToProcess = []
    for files in types:
        filesToProcess.extend(glob.glob(files))
    for f in filesToProcess:
        outPath = f
        mid_term_feature_extractionToFile(f, midTermSize, midTermStep, shortTermSize, shortTermStep, outPath, storeSt_features, storeToCSV, PLOT)
