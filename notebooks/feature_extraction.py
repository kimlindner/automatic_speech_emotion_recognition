# import packages
import os
from pathlib import Path
import re
import progressbar
import math
import numpy as np
import pandas as pd
import sklearn
import librosa
import antropy as ant # for entropy computation
import scipy
from scipy.stats import entropy
from scipy.stats import skew
import parselmouth
from parselmouth.praat import call
import torchaudio
from torchaudio.transforms import LFCC
import warnings

#data_path = os.path.join(str(Path(__file__).parents[1]), 'data/iemocap')
data_path = os.path.join(str(Path(__file__).parents[1]), 'data/ravdess')
#data_path = os.path.join(str(Path(__file__).parents[1]), 'data/emodb/wav')
result_path = os.path.join(str(Path(__file__).parents[1]), 'results')

HOP_LENGTH = 512
FRAME_LENGTH = 2048

# general functions
def frames_gen(y, center=True, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, pad_mode="constant"):
    """
    Pads the audio at the edges of the signal with given pad_mode "constant" as in other librosa features, then
    generates frames with given frame length and hop length. Additionally, the function transposes the resulting array
    to be able to compute other features.
    :param y: librosa audio signal
    :param center: bool; If ``center=True``, the padding mode to use at the edges of the signal. By default, STFT uses
                    zero padding.
    :param frame_length: int; length of a frame (number of signals within one frame)
    :param hop_length: int; length of hopping window, i.e. how much we slide to the side for the next frame
    :param pad_mode: string; which padding mode to use (i.e. 'constant', 'reflect', 'replicate' or 'circular')
    :return: numpy.ndarray; array of size (frame_length, number of frames calculated with hop length and time of audio)
    """
    if y is not None:
        if center:
            padding = [(0, 0) for _ in range(y.ndim)]
            padding[-1] = (int(frame_length // 2), int(frame_length // 2))
            y = np.pad(y, padding, mode=pad_mode)

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

    return np.transpose(frames)

def normalize(x):
    """
    Normalizes a given feature with MinMaxScaler from sklearn.
    :param x: list; feature given as a list
    :return: numpy.ndarray; normalized feature as an array
    """
    return sklearn.preprocessing.MinMaxScaler().fit_transform(np.array(x).reshape(-1,1)).reshape(1,-1)[0]

def average_change_rate(feature, times=None):
    """
    Calcualtes the average change rate of any feature based on the values from consecutive frames
    :param feature: numpy.ndarray; values from a given feature
    :param times: None or numpy.ndarray; if None calculate times with librosa, else use the given times
    :return: numpy.ndarray; average change rate per consecutive frames
    """
    if times is None:
        t = librosa.frames_to_time(range(len(feature)), hop_length=HOP_LENGTH) # convert to time based on feature
    else:
        t = times # if times differ from usual time calculation ###
    avg_change_rate = [] # initialize empty list to track values
    for i in range(len(feature) - 1):
        avg_change_rate.append((feature[i + 1] - feature[i]) / (t[i + 1] - t[i]))

    return np.array(avg_change_rate)

def rising_falling_slopes(feature, times=None):
    """
    Calculates duration of rising and falling slopes of a given feature, i.e. when does the sign of change rate changes?
    :param feature: numpy.ndarray; values from a given feature
    :param times: None or numpy.ndarray; if None calculate times with librosa, else use the given times for average
                    change rate calculation
    :return: tuple; array with duration of rising slopes and array with duration of falling slopes
    """
    if times is None:
        t = librosa.frames_to_time(range(len(feature)), hop_length=HOP_LENGTH) # convert to time based on feature
    else:
        t = times # if times differ from usual time calculation ###
    avg_change_rate = average_change_rate(feature, times)
    duration_rising = []
    duration_falling = []
    value_rising = []
    value_falling = []
    last_t = t[0]
    last_value = feature[0]
    for i, change_rate in enumerate(list(avg_change_rate)):
        if i < len(avg_change_rate) - 1:
            if change_rate > 0 and avg_change_rate[i + 1] < 0:
                duration_rising.append(t[i + 1] - last_t)
                duration_falling.append(0)
                value_rising.append(feature[i + 1] - last_value)
                value_falling.append(0)
                last_t = t[i + 1]
                last_value = feature[i + 1]
            elif change_rate < 0 and avg_change_rate[i + 1] > 0:
                duration_falling.append(t[i + 1] - last_t)
                duration_rising.append(0)
                value_falling.append(feature[i + 1] - last_value)
                value_rising.append(0)
                last_t = t[i + 1]
                last_value = feature[i + 1]
        # in case of arriving at the last element
        else:
            if change_rate > 0:
                duration_rising.append(t[-1] - last_t)
                duration_falling.append(0)
                value_rising.append(feature[-1] - last_value)
                value_falling.append(0)
            else:
                duration_falling.append(t[-1] - last_t)
                duration_rising.append(0)
                value_falling.append(feature[-1] - last_value)
                value_rising.append(0)

    return np.array(duration_rising), np.array(duration_falling), np.array(value_rising), np.array(value_falling)

def energy_comp(y):
    """
    Computes the energy as the sum of squared values within a frame referring to the overall magnitude of the signal.
    :param y: librosa audio signal
    :return: numpy.ndarray; normalized array of energy across the frames
    """
    frames = frames_gen(y) # generate frames
    ener = [np.sum(np.square(frame)) for frame in frames]
    return normalize(ener)

def RMS_energy(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """
    Computes the root mean squared (RMS) values for each frame.
    :param y: librosa audio signal
    :param hop_length: int; length of hopping window, i.e. how much we slide to the side for the next frame
    :param frame_length: int; length of a frame (number of signals within one frame)
    :return: numpy.ndarray; RMS values across the frames
    """
    return librosa.feature.rms(y=y, hop_length=hop_length, frame_length=frame_length)[0]

def RMS_log_entropy(y):
    """
    Calculates the root mean squared (RMS) values of the phase of the signal for each frame.
    :param y: librosa audio signal
    :return: numpy.ndarray; RMS values of phase across the frames
    """
    S, phase = librosa.magphase(librosa.stft(y)) # separate spectrogram in magnitude and phase
    return librosa.feature.rms(S=S)[0]

def amplitude_envelope(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """
    Calculates the envelope of the amplitude of the signal, i.e. the maximal value (amplitude) for each frame.
    :param y: librosa audio signal
    :param frame_length: int; length of a frame (number of signals within one frame)
    :param hop_length: int; length of hopping window, i.e. how much we slide to the side for the next frame
    :return: numpy.ndarray; amplitude values of the signal across the frames
    """
    return np.array([max(y[i:i+frame_length]) for i in range(0, y.size, hop_length)])

def ZCR(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """
    Calculates the zero crossing rate per frame, i.e. how often the signal crosses zero.
    :param y: librosa audio signal
    :param frame_length: int; length of a frame (number of signals within one frame)
    :param hop_length: int; length of hopping window, i.e. how much we slide to the side for the next frame
    :return: numpy.ndarray; zero crossing rates across the frames
    """
    return librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]

def lowlevel_lpc_comp(y, order=4):
    """
    Computes Linear Prediction Coefficients (LPCs) of the given order via Burg’s method on a low level, i.e. for each
    frame.
    :param y: librosa audio signal
    :param order: order of LPC
    :return: numpy.ndarray; of shape (order of LPCs, number of frames)
    """
    frames = frames_gen(y)  # generate frames
    ll_lpcs = [librosa.lpc(frame, order=order) for frame in frames]
    return np.array(ll_lpcs).transpose()

# entropy definitions
def spectral_entropy(y, sr, center=True):
    """
    Computes the spectral entropy of the signal. Spectral Entropy is defined as the Shannon entropy of the power
    spectral density (PSD) of the data:
    math:: H(x, sf) =  -\\sum_{f=0}^{f_s/2} P(f) \\log_2[P(f)]
    Where :math:`P` is the normalised PSD, and :math:`f_s` is the sampling frequency.
    :param y: librosa audio signal
    :param sr: int; ampling rate
    :param center: bool; if true, zero padding is applied for frames generation.
    :return: numpy.ndarray; spectral entropy across frames
    """
    frames = frames_gen(y, center=center)
    with np.errstate(divide='ignore', invalid='ignore'): # ignore division warinings here, will anyways return 0
        spectral = [ant.spectral_entropy(frame, sf=sr, method='welch', normalize=True) for frame in frames]
    return np.array(spectral)

def shannon_entropy(y, base=None):
    """
    Computes the shannon entropy of the signal per frame. Histogram can compute the probability of the occurance of a
    signal value. Entropy computes the Shannon entropy of these probabilities.
    :param y: librosa audio signal
    :param base: the logarithmic base to use, defaults to e
    :return: numpy.ndarray; normalized shannon entropy across frames
    """
    frames = frames_gen(y)
    entropy_contour = [entropy(np.histogram(frame, bins=len(frame), density=True)[0], base=base) for frame in frames]
    return normalize(entropy_contour)

def threshold_entropy(y):
    """
    Computes the threshold entropy of the signal per frame where threshold is the mean of the absolute signal.
    Based on https://www.sciencedirect.com/science/article/pii/S0031320396000659.
    :param y: librosa audio signal
    :return: numpy.ndarray; threshold entropy across frames
    """
    thrd = np.mean(np.abs(y)) # threshold is the mean of the absolute signal

    filtered_signal = np.array([1 if np.abs(val) >= thrd else 0 for val in y])
    frames = frames_gen(filtered_signal)
    thres_ent = [np.mean(frame) for frame in frames]
    return np.array(thres_ent)

def log_energy_entropy(y):
    """
    Computes the logarithmic energy entropy per frame which equals the sum of the logarithm of the square in a frame.
    :param y: librosa audio signal
    :return: numpy.ndarray; normalized logarithmic energy entropy across frames
    """
    frames = frames_gen(y)

    # 0 is ignored for log computation; otherwise we would get distorted results moving between 0 and 1!
    filtered_frames = [frame[frame != 0] for frame in frames]
    log_entropy = np.nan_to_num([np.sum(np.log(np.square(frame))) for frame in filtered_frames])
    return normalize(log_entropy)

def sure_entropy(y, threshold=0.05):
    """
    Computes the sure entropy of the signal per frame.
    Based on https://www.sciencedirect.com/science/article/pii/S0925231216306403.
    :param y: librosa audio signal
    :param threshold: float; threshold with which we need to compare. Chosen heuristically.
    :return: numpy.ndarray; normalized sure entropy across frames
    """
    frames = frames_gen(y)
    sure_ent = []
    for frame in frames:
        hist, bin_edges = np.histogram(frame, bins=len(frame), density=True)
        probs = hist * np.diff(bin_edges)
        sure_ent.append(np.sum(np.minimum(probs, threshold)))

    return normalize(sure_ent)

# fundamental frequency and formants
def f0_comp(y, sr):
    """
    Computes the fundamental frequency of the signal per frame. Fundamental frequency is closely related to pitch,
    which is defined as our perception of fundamental frequency. F0 describes the actual physical phenomenon; pitch
    describes how our ears and brains interpret the signal, in terms of periodicity.
    :param y: librosa audio signal
    :param sr: int; sampling rate
    :return: tuple; array for f0 values across frames, array of boolean indication of voiced parts, array of
    probabilities for indication of voiced parts
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
                                                , hop_length=HOP_LENGTH)
    f0 = np.nan_to_num(f0) # convert nan values to 0
    return f0, voiced_flag, voiced_prob

def pitch_comp(y):
    """
    Computes the pitch of the signal using praat. Pitch describes how our ears and brains interpret the signal in terms
    of periodicity.
    :param y: praat audio signal
    :return: tuple; array of pitch values, array of pitch times according to praat
    """
    pitch = y.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_time = pitch.xs()
    return pitch_values, pitch_time

def  formant_analysis(y, gender, formant_order=4, f0min = 75, f0max = 600):
    """
    Computes the formants of the signal up to the given order with praat.
    :param y: praat audio signal
    :param gender: string; either male or female, determines the maximal frequency according to the speaker's gender
    :param formant_order: int; order up to which we want to calculate the formants
    :param f0min: int; the standard is 75
    :param f0max: int; the standard is 600 (https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__ac____.html).
    :return: dictionary; key and values for the different formants and their statistics (median, maximum, mean, standard
    deviation, variance, average change rate)
    """
    # compute the occurrences of periodic instances in the signal
    pointProcess = call(y, "To PointProcess (periodic, cc)", f0min, f0max)

    """
    ### added in second round of feature generation for female / male differentiation
    # define maximal frequency depending on gender
    if gender == 'f':
        formant_ceiling = 5500
    else:
        formant_ceiling = 5000
    """
    formant_ceiling = 5000 # decided for no differentiation female/male because we achieved better results

    formants = call(y, "To Formant (burg)", 0.0025, 5, formant_ceiling, 0.025, 50)  # formants definition

    # assign formant values with times where they make sense (periodic instances)
    numPoints = call(pointProcess, "Get number of points")

    # define a dictionary to save the formants of the specified order and initialize with empty list
    form_dict = {}
    for order in range(1, formant_order + 1):
        # initialize formant dictionary with keys for each formant
        form_dict['f' + str(order)] = []

        for point in range(0, numPoints):
            point += 1
            t = call(pointProcess, "Get time from index", point)
            formant = call(formants, "Get value at time", order, t, 'Hertz', 'Linear')
            if str(formant) != 'nan':
                form_dict['f' + str(order)].append(formant)

        # calculate statistics of the given formants
        # if praat does not find suitable num points, the formants will be empty. In order to keep automatic feature
        # extraction, we can overwrite these values with nan and exclude the entries later on in preprocessing.
        if form_dict['f' + str(order)] != []:
            form_dict['f' + str(order) + '_median'] = np.median(form_dict['f' + str(order)])
            form_dict['f' + str(order) + '_max'] = np.max(form_dict['f' + str(order)])
            form_dict['f' + str(order) + '_mean'] = np.mean(form_dict['f' + str(order)])
            form_dict['f' + str(order) + '_std'] = np.std(form_dict['f' + str(order)])
            form_dict['f' + str(order) + '_var'] = np.var(form_dict['f' + str(order)])
            form_dict['f' + str(order) + '_avg_change_rate'] = average_change_rate(form_dict['f' + str(order)])
        else:
            # in some files of iemocap the automatic formant extraction with praat is not working; hence we will assign
            # nan values here and exclude them later
            form_dict['f' + str(order)] = np.nan
            form_dict['f' + str(order) + '_median'] = np.nan
            form_dict['f' + str(order) + '_max'] = np.nan
            form_dict['f' + str(order) + '_mean'] = np.nan
            form_dict['f' + str(order) + '_std'] = np.nan
            form_dict['f' + str(order) + '_var'] = np.nan
            form_dict['f' + str(order) + '_avg_change_rate'] = np.nan

    return form_dict

# speed of speech
def speech_rate(sound):
    """
    Estimates the speaking rate, articulation rate and ASD without text conversion.
    Based on "Praat Script Syllable Nuclei" from De Jong, N.H. & Wempe, T. (2009). Praat script to detect syllable
    nuclei and measure speech rate automatically. Behavior research methods, 41 (2), 385 - 390.
    Updated by Hugo Quené, Ingrid Persoon, & Nivja de Jong in 2017 and translated to Python with Parselmouth by David
    Feinberg in 2019.
    Only slight changes made. It seems sufficient to calculate the overall rates for emotion detection and not per frame.
    :param sound: praat audio signal
    :return: dictionary; contains number of voiced syllables, number of pauses, original duration, intensity duration,
    speaking rate, articulation rate, and average syllable duration (asd)
    """
    silencedb = -25
    mindip = 2
    minpause = 0.3
    originaldur = sound.get_total_duration()
    intensity = sound.to_intensity(50)
    start = call(intensity, "Get time from frame number", 1)
    nframes = call(intensity, "Get number of frames")
    end = call(intensity, "Get time from frame number", nframes)
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

    # get .99 quantile to get maximum (without influence of non-speech sound bursts)
    max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

    # estimate Intensity threshold
    threshold = max_99_intensity + silencedb
    threshold2 = max_intensity - max_99_intensity
    threshold3 = silencedb - threshold2
    if threshold < min_intensity:
        threshold = min_intensity

    # get pauses (silences) and speakingtime
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # if the difference between loudest and softest sound is too low, we'll get a warning here which we want to
        # suprress for now. Anyways, we will only use features that are relevant for the models later on. We can
        # therefore still keep the feature even if the difference is low.
        textgrid = call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")
    silencetier = call(textgrid, "Extract tier", 1)
    silencetable = call(silencetier, "Down to TableOfReal", "sounding")
    npauses = call(silencetable, "Get number of rows")
    speakingtot = 0
    for ipause in range(npauses):
        pause = ipause + 1
        beginsound = call(silencetable, "Get value", pause, 1)
        endsound = call(silencetable, "Get value", pause, 2)
        speakingdur = endsound - beginsound
        speakingtot += speakingdur

    intensity_matrix = call(intensity, "Down to Matrix")
    # sndintid = sound_from_intensity_matrix
    sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
    # use total duration, not end time, to find out duration of intdur (intensity_duration)
    # in order to allow nonzero starting times.
    intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
    intensity_max = call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
    point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    # estimate peak positions (all peaks)
    numpeaks = call(point_process, "Get number of points")
    t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

    # fill array with intensity values
    timepeaks = []
    peakcount = 0
    intensities = []
    for i in range(numpeaks):
        value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
        if value > threshold:
            peakcount += 1
            intensities.append(value)
            timepeaks.append(t[i])

    # fill array with valid peaks: only intensity values if preceding
    # dip in intensity is greater than mindip
    validpeakcount = 0
    currenttime = timepeaks[0]
    currentint = intensities[0]
    validtime = []

    for p in range(peakcount - 1):
        following = p + 1
        followingtime = timepeaks[p + 1]
        dip = call(intensity, "Get minimum", currenttime, followingtime, "None")
        diffint = abs(currentint - dip)
        if diffint > mindip:
            validpeakcount += 1
            validtime.append(timepeaks[p])
        currenttime = timepeaks[following]
        currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

    # Look for only voiced parts
    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedcount = 0
    voicedpeak = []

    for time in range(validpeakcount):
        querytime = validtime[time]
        whichinterval = call(textgrid, "Get interval at time", 1, querytime)
        whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(querytime)
        if not math.isnan(value):
            if whichlabel == "sounding":
                voicedcount += 1
                voicedpeak.append(validtime[time])

    # calculate time correction due to shift in time for Sound object versus
    # intensity object
    timecorrection = originaldur / intensity_duration

    # Insert voiced peaks in TextGrid
    call(textgrid, "Insert point tier", 1, "syllables")
    for i in range(len(voicedpeak)):
        position = (voicedpeak[i] * timecorrection)
        call(textgrid, "Insert point", 1, position, "")

    # return results
    speakingrate = voicedcount / originaldur
    articulationrate = voicedcount / speakingtot
    npause = npauses - 1
    if voicedcount != 0:
        asd = speakingtot / voicedcount
    else:
        asd = 0
    speechrate_dictionary = {'nsyll':voicedcount,
                             'npause': npause,
                             'dur(s)':originaldur,
                             'phonationtime(s)':intensity_duration,
                             'speechrate(nsyll / dur)': speakingrate,
                             'articulation rate(nsyll / phonationtime)':articulationrate,
                             'ASD(speakingtime / nsyll)':asd}
    return speechrate_dictionary

# brightness
def spectral_centroid(y, sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """
    Computes the spectral centroid as the brightness of the signal per frame.
    Spectral centroid is the centre of gravity of the magnitude spectrum; frequency band where most of the energy is
    concentrated. It measures the brightness of the sound (see https://github.com/kimlindner/AudioSignalProcessingForML/)
    :param y: librosa audio signal
    :param sr: int; sample rate
    :param frame_length: int; length of a frame (number of signals within one frame)
    :param hop_length: int; length of hopping window, i.e. how much we slide to the side for the next frame
    :return: numpy.ndarray, normalized spectral centroid / brightness per frame
    """
    spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    return normalize(spectral_cent)

# mfccs
def mfcc_comp(y, sr, n):
    """
    Computes n Mel Frequency Cepstrum Coefficients (MFCCs), its first, and second derivative with librosa.
    :param y: librosa audio signal
    :param sr: int; sample rate
    :param n: int; number of MFCCs to calculate
    :return: tuple; with standard, first, and second order derivative mfccs, each contains a (n,feature length))-
    dimensional array with n entries per short-frame
    """
    mfccs = librosa.feature.mfcc(y=y, n_mfcc=n, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    return mfccs, delta_mfccs, delta2_mfccs

# lpccs
def lpcc(y, lpc_order, cepsorder=None):
    """
    Computes Linear Prediction Cepstrum Coefficients (LPCCs) of the given signal.
    Code partly from https://www.kaggle.com/code/sourabhy/working-with-speech.
    Code based on algorithm from Rao et al. (2015).Language Identification Using Spectral and Prosodic Features.
    (https://link.springer.com/content/pdf/bbm:978-3-319-17163-0/1.pdf) &
    Al-Alaoui et al. (2008). Speech Recognition using Artificial Neural Networks and Hidden Markov Models.
    (https://feaweb.aub.edu.lb/research/dsaf/Publications/IMCL114.pdf) &
    Matlab ressouce https://de.mathworks.com/help/dsp/ref/lpctofromcepstralcoefficients.html.
    :param y: librosa audio signal
    :param lpc_order: int; LPC coefficients order
    :param cepsorder: int; LPCC coefficients order
    :return: list; LPCC coefficients for the given sequence y
    """
    # compute LPC coefficients
    coefs = librosa.lpc(y, order=lpc_order)

    # compute error term with librosa source code
    b = np.hstack([[0], -1 * coefs[1:]])
    y_hat = scipy.signal.lfilter(b, [1], y)
    err_term = np.sum(np.square(y - y_hat))

    # compute LPCCs
    if cepsorder is None:
        cepsorder = len(coefs) - 1

    lpcc_coeffs = [np.log(err_term) if err_term != 0 else coefs[0]] # lpcc coeffs for i=0,1
    for n in range(1, cepsorder): # lpcc coeffs for i=1,...,order-1 of lpccs
        # Use order as upper bound for the last iteration (want 0,...,order-1 lpccs)
        upbound = (cepsorder if n > cepsorder - 1 else n)
        lpcc_coef = sum(i * lpcc_coeffs[i] * coefs[n - i]
                         for i in range(1, upbound)) * 1. / upbound # sum over upper bound - 1
        lpcc_coef += coefs[n] if n <= len(coefs) else 0 # for both cases (m<p and m>p)
        lpcc_coeffs.append(lpcc_coef)

    return lpcc_coeffs

def lowlevel_lpcc_comp(y, lpc_order, cepsorder=None):
    """
    Computes LPCCs on a low level, i.e. LPCCs of order cepsorder for each frame.
    :param y: librosa audio signal
    :param cepsorder: int; order of cepstrum
    :return: numpy.ndarray; LPCCs of order cepsorder per frame, has shape (cepsorder, number of frames)
    """
    frames = frames_gen(y)  # generate frames
    ll_lpccs = [lpcc(frame, lpc_order, cepsorder=cepsorder) for frame in frames]
    return np.array(ll_lpccs).transpose()

# lpcmfcc
def lpcmfcc_comp(lpccs, alpha=0.35, order_n=None):
    """
    Computes Linear Prediction Coefficients and Mel Frequency Cepstrum Coefficients (LPCMFCCs) based on the given LPCC
    coefficients recursively with order_n iterations going down. Computation mainly based on
    http://www.ecice06.com/EN/abstract/abstract9872.shtml.
    :param lpccs: list of size cepsorder; computed LPCC coefficients
    :param alpha: float; usually between 0.31 and 0.35 to be close to Mel scale
    :param order_n: int; number of iterations, generally equal to order of LPCCs
    :return: list of size cepsorder; LPCMFCC coefficients from last iteration at n=0
    """
    order_k = len(lpccs)
    if order_n == None:
        order_n = order_k

    # initialize MCs(n) at iteration n = k (use order - 1 because we are going from 0)
    melcep_coefs = [lpccs[order_n - 1]]
    melcep_coefs.extend(np.zeros(order_k - 1))

    for n in range(order_n - 2, -1, -1):  # order_n iterations going down to 0
        melcep_coefs_old = melcep_coefs.copy()
        melcep_coefs[0] = lpccs[n] + alpha * melcep_coefs_old[0]
        melcep_coefs[1] = (1 - alpha) ** 2 * melcep_coefs_old[0] + alpha * melcep_coefs_old[1]
        for k in range(2, order_k):  # order_k coefficients to compute
            melcep_coefs[k] = melcep_coefs_old[k - 1] + alpha * (melcep_coefs_old[k] - melcep_coefs[k - 1])

    return melcep_coefs

def lowlevel_lpcmfcc_comp(lpccs_local, alpha=0.35, order_n=None):
    """
    Computes LPCMFCCs on a low level, i.e. taking the computed LPCCs from each frame and computing LPCMFCCs for that
    frame
    :param lpccs_local: numpy array of shape (cepsorder, number of frames); all LPCCs per frame
    :param alpha: float; usually between 0.31 and 0.35 to be close to Mel scale
    :param order_n: int; number of iterations, generally equal to order of LPCCs
    :return: numpy.ndarray; of shape (cepsorder, number of frames); all LPCMFCCs per frame
    """
    ll_lpcmfccs = [lpcmfcc_comp(lpccs_frame, alpha=alpha, order_n=order_n) for lpccs_frame in lpccs_local.transpose()]
    return np.array(ll_lpcmfccs).transpose()

# lfccs
def lfcc_comp(y, sr, n_lfcc, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """
    Computes LFCCs (Linear Frequency Cepstral Coefficients) with torchaudio, i.e. spectral energy dynamic coefficients
    of equally spaced frequency bands. MFCCs are calculated on the Mel scale using Melspectrogram but LFCCs are based on
    Spectrogram.
    :param y: torchaudio audio signal
    :param sr: int; sampling rate
    :param n_lfcc: int; order of LFCCs, i.e. number of coefficients to compute
    :param frame_length: int; length of a frame (number of signals within one frame)
    :param hop_length: int; length of hopping window, i.e. how much we slide to the side for the next frame
    :return: numpy.ndarray of size (n_lfcc, number of frames); LFCC coefficients per frame
    """
    lfccs = LFCC(sample_rate=sr, n_lfcc=n_lfcc, speckwargs={"n_fft": frame_length, "hop_length": hop_length,
                                                                 'pad_mode': 'constant'})(y)
    return lfccs.numpy()[0] # convert tensor to numpy array

def feature_extraction(filename, path, database, file_label):
    """
    Extracts all necessary features per audio file.
    :param filename: name of the file
    :param path: path in which the file lies
    :param database: name of the database used
    :param label: string; label for iemocap already extracted; none for other databases
    :return: dictionary; with file name, label, and all computed features.
    """
    if database == 'emodb':
        audio_path = os.path.join(path, filename)

        # label and speaker information for emo db
        label_dict = {'W': 'anger', 'L': 'boredom', 'E': 'disgust', 'A': 'fear', 'F': 'happiness', 'T': 'sadness',
                      'N': 'neutral'}
        label = label_dict[filename[5]]
        speaker_num = filename[:2]
        gender_dict = {'03': 'm', '08': 'f', '09': 'f', '10': 'm', '11': 'm', '12': 'm', '13': 'f', '14': 'f',
                       '15': 'm',
                       '16': 'f'}
        gender = gender_dict[speaker_num]

    elif database == 'ravdess':
        audio_path = filename
        filename = filename.split('\\')[-1]

        # label and speaker information for ravdess
        filename_list = filename.split('-')
        label_dict = {'01': 'neutral', '02': 'calmness', '03': 'happiness', '04': 'sadness', '05': 'anger',
                      '06': 'fear', '07': 'disgust', '08': 'surprise'}
        label = label_dict[filename_list[2]]
        speaker_num = filename_list[-1].split('.')[0]
        gender = 'm' if int(speaker_num) % 2 != 0 else 'f'

    elif database == 'iemocap':
        audio_path = filename
        filename = filename.split('\\')[-1]

        # label and speaker information for iemocap
        label_dict = {'ang':'anger', 'dis':'disgust', 'fea':'fear', 'hap':'happiness', 'neu':'neutral', 'sad':'sadness'}
        label = label_dict[file_label]
        speaker_num = filename[3:5]
        gender = filename[5].lower()

    y, sr = librosa.load(audio_path, sr=None)  # load librosa audio
    y_praat = parselmouth.Sound(audio_path)  # load praat audio
    y_torch, sr_torch = torchaudio.load(audio_path)  # load torch audio

    # only extract data for the chosen 7 emotions
    if label in ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']:

        duration = librosa.get_duration(y=y, sr=sr)

        # basic statistics of signal
        mean = np.mean(np.abs(y))
        median = np.median(np.abs(y))
        max = np.max(y)
        min = np.min(y)
        var = np.var(y)
        std = np.std(y)

        zcr = ZCR(y)
        energy = energy_comp(y)
        energy_avg_change_rate = average_change_rate(energy)
        duration_rising_energy, duration_falling_energy, value_rising_energy, value_falling_energy = rising_falling_slopes(energy)
        rms_energy = RMS_energy(y)
        log_rms = RMS_log_entropy(y)
        amplitude = amplitude_envelope(y)
        amplitude_avg_change_rate = average_change_rate(amplitude)
        lpc_global = librosa.lpc(y, order=4)
        lpc_local = lowlevel_lpc_comp(y)


        # entropy computations
        spectral_ent = spectral_entropy(y, sr)
        log_energy_ent = log_energy_entropy(y)
        shannon_ent = shannon_entropy(y)
        threshold_ent = threshold_entropy(y)
        sure_ent = sure_entropy(y)

        # f0, formants
        f0, voiced_flag, voiced_prob = f0_comp(y, sr)
        f0_avg_change_rate = average_change_rate(f0)
        formants = formant_analysis(y_praat, gender)

        # pitch
        pitch_values, pitch_time = pitch_comp(y_praat)
        duration_rising_pitch, duration_falling_pitch, value_rising_pitch, value_falling_pitch = rising_falling_slopes(
            pitch_values, times=pitch_time)

        # speaking rate, articulation rate, asd
        speaking_dictionary = speech_rate(y_praat)
        speaking_rate = speaking_dictionary['speechrate(nsyll / dur)']
        articulation_rate = speaking_dictionary['articulation rate(nsyll / phonationtime)']
        asd = speaking_dictionary['ASD(speakingtime / nsyll)']

        # brightness (spectral centroid)
        spectral_cent = spectral_centroid(y, sr)

        # mfccs, lpccs, lpcmfccs, lfccs
        mfccs, delta_mfccs, delta2_mfccs = mfcc_comp(y, sr, n=13)
        lpccs_global = lpcc(y, lpc_order=12, cepsorder=12)
        lpccs_local = lowlevel_lpcc_comp(y, lpc_order=12, cepsorder=12)
        lpcmfccs_global = lpcmfcc_comp(lpccs_global)
        lpcmfccs_local = lowlevel_lpcmfcc_comp(lpccs_local)
        lfccs = lfcc_comp(y_torch, sr_torch, n_lfcc=12)

        features_dict = {'file': filename, 'label': label, 'speaker': speaker_num, 'gender': gender, 'duration': duration,
                         'mean': mean, 'median': median, 'max': max, 'min': min, 'var': var, 'std': std, 'zcr': zcr,
                         'energy': energy, 'energy_avg_change_rate': energy_avg_change_rate,
                         'duration_rising_energy': duration_rising_energy,
                         'duration_falling_energy': duration_falling_energy, 'value_rising_energy': value_rising_energy,
                         'value_falling_energy': value_falling_energy, 'rms': rms_energy, 'log_rms': log_rms,
                         'amplitude': amplitude, 'amplitude_avg_change_rate': amplitude_avg_change_rate,
                         'lpc_global': lpc_global,
                         'lpc_local': lpc_local,
                         'spectral_entropy': spectral_ent, 'shannon_entropy': shannon_ent,
                         'threshold_entropy': threshold_ent,
                         'log_energy_entropy': log_energy_ent, 'sure_entropy': sure_ent, 'f0': f0, 'voiced': voiced_flag,
                         'f0_avg_change_rate': f0_avg_change_rate, 'pitch': pitch_values, 'pitch_time': pitch_time,
                         'duration_rising_pitch': duration_rising_pitch, 'duration_falling_pitch': duration_falling_pitch,
                         'value_rising_pitch': value_rising_pitch, 'value_falling_pitch': value_falling_pitch,
                         'speaking_rate': speaking_rate, 'articulation_rate': articulation_rate, 'asd': asd,
                         'spectral_centroid': spectral_cent, 'mfccs': mfccs, 'delta_mfccs': delta_mfccs,
                         'delta2_mfccs': delta2_mfccs, 'lpccs_local': lpccs_local, 'lpccs_global': lpccs_global,
                         'lpcmfccs_global': lpcmfccs_global, 'lpcmfccs_local': lpcmfccs_local, 'lfccs': lfccs} | formants

        return features_dict
    else:
        return None

def run_all_files(data_path, result_path, result_name, database):
    """
    Runs all files for feature extraction within the given path and saves a dataframe to the result path.
    :param data_path: path where the audio files are
    :param result_path: path where the dataframe should be saved in a pickle file
    :param result_name: name of the resulting dataframe
    :return: dataframe with all files and features extracted
    """
    print('Start running for {}.'.format(database))

    if database == 'emodb':
        audio_files = os.listdir(data_path)
    elif database == 'ravdess':
        audio_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                audio_files.append(os.path.join(root, file))
    elif database == 'iemocap':
        audio_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                audio_files.append(os.path.join(root, file))

        reg = re.compile(r'^(?=.*sentences)(?=.*script)(?!.*ForcedAlignment)(?=.*wav).*$')
        audio_files = [file for file in audio_files if re.search(reg, file)]

        # get the label for each audio file and save them in a dictionary
        # (since label file contains label for several files)
        label_files = []
        for audio_file in audio_files:
            file_name = audio_file.split('\\')[-1][:-9]
            label_files.append(audio_file.split('\\sentences')[0] + '\dialog\EmoEvaluation\\'  + file_name + '.txt')

        label_files = sorted(list(set(label_files))) # only get unique files
        script_label_dict = {}
        for label_file in label_files:
            try:
                with open(label_file) as f:
                    lines = f.readlines()
                lines_filtered = [line for line in lines if 'script' in line]
                for line in lines_filtered:
                    line_split = line.split('\t')
                    script_label_dict[line_split[1]] = line_split[2]
            except:
                continue

        # exclude files where the label was not unique, i.e. where the label on utterance level is given as 'xxx', as
        # well as where the label is not coherent with the labels we are working with in emodb, i.e., excitement,
        # frustration, other, and surprise
        filtered_label_dict = {key: value for key, value in script_label_dict.items() if value not in ['xxx', 'exc', 'fru', 'oth', 'sur']}
        valid_files = list(filtered_label_dict.keys())
        # (additionally exclude '.pk' since there was one pickle file in session 3 which is not fitting to the rule)
        audio_files = [file for file in audio_files if
                       file.split('\\')[-1].split('.')[0] in valid_files and '.pk' not in file]


    # create an final list to store all results and convert to dataframe
    final_list = []

    i = 0
    with progressbar.ProgressBar(max_value=len(audio_files)) as bar:
        for file in audio_files:
            if database == 'iemocap':
                # need to extract filename for iemocap in order to get label beforehand since it is saved in a different
                # file for several audios
                file_name = file.split('\\')[-1].split('.')[0]
                label = filtered_label_dict[file_name]
            else:
                label = None
            feature_dict = feature_extraction(file, data_path, database, label)
            if feature_dict != None:
                final_list.append(feature_dict)
            i += 1
            bar.update(i)


    df = pd.DataFrame(final_list)
    df.to_pickle(os.path.join(result_path, result_name))
    print('Features extracted. File written to {}'.format(result_name))

    return df


def finalize_features(result_path, input_name, result_name):
    """
    Finalizes the feature extraction part with some statistical calculations for the different features and writes the
    newly updated dataframe in a new pickle file.
    :param result_path: path where results are stored
    :param input_name: name of the created feature extraction file
    :param result_name: name of the modified feature extraction file
    :return: finalized dataframe
    """
    from warnings import simplefilter
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    print('Modify extracted features by adding statistical calculations.')
    df = pd.read_pickle(os.path.join(result_path, input_name))

    # general statistics for all list or array type features
    print('Calculate general statistics.')

    # first clear rising and falling slopes from 0 values
    for feature in ['energy', 'pitch']:
        for elem in ['duration', 'value']:
            # clear lists from 0 values
            df[elem + '_rising_' + feature] = df[elem + '_rising_' + feature].apply(lambda x: x[x != 0])
            df[elem + '_falling_' + feature] = df[elem + '_falling_' + feature].apply(lambda x: x[x != 0])

    # calculate statistics for all
    for feature in list(df.columns):
        if (type(df[feature][0]) == list) or (type(df[feature][0]) == np.ndarray and df[feature][0].ndim == 1):


            # we have to make sure that the feature row is not an empty list or nan from the previous calculation
            df[feature + '_max'] = df[feature].apply(lambda x: np.max(x) if (type(x) == list or type(x) == np.ndarray)
                                                                            and len(x) != 0 else np.nan)
            df[feature + '_min'] = df[feature].apply(lambda x: np.min(x) if (type(x) == list or type(x) == np.ndarray)
                                                                            and len(x) != 0 else np.nan)
            df[feature + '_mean'] = df[feature].apply(lambda x: np.mean(x) if (type(x) == list or type(x) == np.ndarray)
                                                                              and len(x) != 0 else np.nan)
            df[feature + '_median'] = df[feature].apply(lambda x: np.median(x) if (type(x) == list or type(x) == np.ndarray)
                                                                                  and len(x) != 0 else np.nan)
            df[feature + '_var'] = df[feature].apply(lambda x: np.var(x) if (type(x) == list or type(x) == np.ndarray)
                                                                            and len(x) != 0 else np.nan)

        elif (type(df[feature][0]) == np.ndarray and df[feature][0].ndim == 2):
            # calculate statistics of cepstrum coefficients mfccs, lpccs, lpcmfccs, lpcs,..., i.e. multidimensional arrays
            for index, row in df.iterrows():
                for i, coef in enumerate(row[feature]):
                    # create mean, variance, maximum, and minumum of all MFCCs/LPCCs/LPCMFCCs respectively
                    df.loc[index, feature + str(i) + '_max'] = np.max(coef)
                    df.loc[index, feature + str(i) + '_min'] = np.min(coef)
                    df.loc[index, feature + str(i) + '_mean'] = np.mean(coef)
                    df.loc[index, feature + str(i) + '_median'] = np.median(coef)
                    df.loc[index, feature + str(i) + '_var'] = np.var(coef)

    # calculate statistics for energy and pitch with rising/falling slopes
    print('Calculate energy and pitch statistics.')
    statistics_features = ['energy', 'pitch']
    for feature in statistics_features:
        for elem in ['duration', 'value']:
            # iqr for rising slopes of feature
            df[elem + '_rising_' + feature + '_iqr'] = df[elem + '_rising_' + feature].apply(lambda x: np.subtract(
                *np.percentile(x, [75, 25])) if (type(x) == list or type(x) == np.ndarray) and len(x) != 0 else np.nan)

            # iqr for falling slopes of feature
            df[elem + '_falling_' + feature + '_iqr'] = df[elem + '_rising_' + feature].apply(lambda x: np.subtract(
                *np.percentile(x, [75, 25])) if (type(x) == list or type(x) == np.ndarray) and len(x) != 0 else np.nan)

    # further stats for pitch, energy
    df['pitch_non0'] = df['pitch'].apply(lambda x: x[x != 0])
    df['skew_log_pitch'] = df['pitch_non0'].apply(
        lambda x: skew(np.log(x)) if (type(x) == list or type(x) == np.ndarray) and len(x) != 0 else np.nan)
    df['range_log_pitch'] = df['pitch_non0'].apply(
        lambda x: np.abs((np.max(x) - np.min(x))) if (type(x) == list or type(x) == np.ndarray) and len(
            x) != 0 else np.nan)
    df['energy_non0'] = df['energy'].apply(lambda x: x[x != 0])
    df['range_log_energy'] = df['energy_non0'].apply(
        lambda x: np.abs((np.max(x) - np.min(x))) if (type(x) == list or type(x) == np.ndarray) and len(
            x) != 0 else np.nan)
    df.drop(columns=['pitch_non0', 'energy_non0'], inplace=True)

    # remove columns that only have the same values (e.g. happens for energy_max for example)
    print('Remove columns that only have the same values.')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    mask = (df[num_cols].nunique() == 1) # get a boolean mask of columns with only one unique value
    cols_to_drop = mask[mask].index # get the column names to drop
    df.drop(cols_to_drop, axis=1, inplace=True) # drop the columns

    df.to_pickle(os.path.join(result_path, result_name))
    print('Modified file written to {}.'.format(result_name))

    return df

"""file = r'C:\\Users\\Kim-Carolin\\Documents\\GitHub\\automatic_speech_emotion_recognition\\data/iemocap\\IEMOCAP_full_release\\Session1\\sentences\\wav\\Ses01M_script02_1\\Ses01M_script02_1_F021.wav'
file_name = file.split('\\')[-1].split('.')[0]
label = 'neu'
feature_extraction(file, data_path, 'iemocap', label)"""

run_all_files(data_path=data_path, result_path=result_path, result_name='extracted_features_ravdess_check.pkl', database='ravdess')
finalize_features(result_path=result_path, input_name='extracted_features_ravdess_check.pkl',
                  result_name='extracted_features_modified_all_stats_ravdess_check.pkl')
