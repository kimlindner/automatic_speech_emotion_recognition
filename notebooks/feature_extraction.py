# import packages
import os
from pathlib import Path
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
from parselmouth import praat
from parselmouth.praat import call


data_path = os.path.join(str(Path(__file__).parents[1]), 'data/wav')
result_path = os.path.join(str(Path(__file__).parents[1]), 'results')

HOP_LENGTH = 512
FRAME_LENGTH = 2048

# general functions
def frames_gen(y, center=True, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, pad_mode="constant"):
    """
    generates frames like in librosa source code, additionally transposes array
    :param y: audio signal
    :param center:
    :param frame_length:
    :param hop_length:
    :param pad_mode:
    :return:
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

    :param x:
    :return:
    """
    return sklearn.preprocessing.MinMaxScaler().fit_transform(np.array(x).reshape(-1,1)).reshape(1,-1)[0]

def average_change_rate(feature, times=None):
    """
    calcualtes the average change rate of any feature based on the values form consecutive frames
    :param feature: numpy array, values from a given feature
    :return: numpy array, average change rate per consecutive frames
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
    function that calculates duration of rising and falling slopes of a given feature, i.e. when does the sign of change
    rate changes?
    :param feature: numpy array, any feature calculated on the data set
    :return: tuple, array with duration of rising slopes and array with duration of falling slopes
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

    :param y:
    :return:
    """
    frames = frames_gen(y) # generate frames
    ener = [np.sum(np.square(frame)) for frame in frames]
    return normalize(ener)

def RMS_energy(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """

    :param y:
    :param hop_length:
    :param frame_length:
    :return:
    """
    return librosa.feature.rms(y=y, hop_length=hop_length, frame_length=frame_length)[0]

def RMS_log_entropy(y):
    """

    :param y:
    :return:
    """
    S, phase = librosa.magphase(librosa.stft(y)) # separate spectrogram in magnitude and phase
    return librosa.feature.rms(S=S)[0]

def amplitude_envelope(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """

    :param y:
    :param frame_length:
    :param hop_length:
    :return:
    """
    return np.array([max(y[i:i+frame_length]) for i in range(0, y.size, hop_length)])

def ZCR(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    """

    :param y:
    :param frame_length:
    :param hop_length:
    :return:
    """
    return librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]

def lpc_est(y, order=4):
    """

    :param y:
    :param order:
    :return:
    """
    return librosa.lpc(y, order=order)

def lpcc_est(y, order=12):
    """
    LPCCs are coefficients obtained by applying Fourier transformation on the logarithmic magnitude spectrum of LPC.
    :param y:
    :param order:
    :return:
    """

# entropy definitions
def spectral_entropy(y, sr, center=True):
    """
    Spectral Entropy is defined to be the Shannon entropy of the power spectral density (PSD) of the data:
    math:: H(x, sf) =  -\\sum_{f=0}^{f_s/2} P(f) \\log_2[P(f)]
    Where :math:`P` is the normalised PSD, and :math:`f_s` is the sampling frequency.
    :param y:
    :param sf:
    :param center:
    :return:
    """
    frames = frames_gen(y, center=center)
    spectral = [ant.spectral_entropy(frame, sf=sr, method='welch', normalize=True) for frame in frames]
    return np.array(spectral)

def shannon_entropy(y, base=None):
    """

    :param y:
    :param base:
    :return:
    """
    frames = frames_gen(y)
    entropy_contour = [entropy(np.histogram(frame, bins=len(frame), density=True)[0], base=base) for frame in frames]
    return normalize(entropy_contour)

def threshold_entropy(y):
    """

    :param y:
    :return:
    """
    thrd = np.mean(np.abs(y)) # threshold is the mean of the absolute signal

    filtered_signal = np.array([1 if np.abs(val) >= thrd else 0 for val in y])
    frames = frames_gen(filtered_signal)
    thres_ent = [np.mean(frame) for frame in frames]
    return np.array(thres_ent)

def log_energy_entropy(y):
    """

    :param y:
    :return:
    """
    frames = frames_gen(y)

    # 0 is ignored for log computation; otherwise we would get distorted results moving between 0 and 1!
    filtered_frames = [frame[frame != 0] for frame in frames]
    log_entropy = np.nan_to_num([np.sum(np.log(np.square(frame))) for frame in filtered_frames])
    return normalize(log_entropy)

def sure_entropy(y, threshold=0.05):
    """

    :param y:
    :param threshold:
    :return:
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
     Fundamental frequency is closely related to pitch, which is defined as our perception of fundamental frequency.
     F0 describes the actual physical phenomenon; pitch describes how our ears and brains interpret the signal, in terms
     of periodicity.
    :param y:
    :param sr:
    :return:
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), hop_length=HOP_LENGTH)
    f0 = np.nan_to_num(f0) # convert nan values to 0
    return f0, voiced_flag, voiced_prob

def pitch_comp(y):
    """
    Fundamental frequency is closely related to pitch, which is defined as our perception of fundamental frequency.
    F0 describes the actual physical phenomenon; pitch describes how our ears and brains interpret the signal, in terms
    of periodicity.
    :param y:
    :return:
    """
    pitch = y.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_time = pitch.xs()
    return pitch_values, pitch_time

def formant_analysis(y, gender, formant_order=4, f0min = 75, f0max = 600):
    """
    f0_max: in the example it was 300; but here we see standard is 600
    (https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__ac____.html)
    :param filename:
    :param path:
    :param gender:
    :param formant_order:
    :param f0min:
    :param f0max:
    :return:
    """

    # compute the occurrences of periodic instances in the signal
    pointProcess = call(y, "To PointProcess (periodic, cc)", f0min, f0max)

    # define maximal frequency depending on gender
    if gender == 'female':
        formant_ceiling = 5500
    else:
        formant_ceiling = 5000

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

        # calculate median, max, and mean of the given formants
        form_dict['f' + str(order) + '_median'] = np.median(form_dict['f' + str(order)])
        form_dict['f' + str(order) + '_max'] = np.max(form_dict['f' + str(order)])
        form_dict['f' + str(order) + '_mean'] = np.mean(form_dict['f' + str(order)])
        form_dict['f' + str(order) + '_std'] = np.std(form_dict['f' + str(order)])
        form_dict['f' + str(order) + '_var'] = np.var(form_dict['f' + str(order)])
        form_dict['f' + str(order) + '_avg_change_rate'] = average_change_rate(form_dict['f' + str(order)])

        ## rather together in 1 feature?


        # drop key with all formant numbers
        form_dict.pop('f' + str(order), None)

    return form_dict

# speed of speech
def speech_rate(sound):
    """
    Function that estimates speaking rate and articulation rate and ASD without text conversion.
    Based on "Praat Script Syllable Nuclei" from De Jong, N.H. & Wempe, T. (2009). Praat script to detect syllable
    nuclei and measure speech rate automatically. Behavior research methods, 41 (2), 385 - 390.
    Updated by Hugo Quené, Ingrid Persoon, & Nivja de Jong in 2017 and translated to Python with Parselmouth by David
    Feinberg in 2019.
    Only slight changes made.
    :param sound: sound loaded with praat
    :return: dictionary containing number of voiced syllables, number of pauses, original duration, intensity duration,
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
    asd = speakingtot / voicedcount
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
    Spectral centroid is the centre of gravity of the magnitude spectrum; frequency band where most of the energy is
    concentrated. It measures the brightness of the sound (see https://github.com/kimlindner/AudioSignalProcessingForML/)
    :param y: audio file loaded with librosa
    :param sr: int, sample rate
    :param frame_length: int, length of a frame
    :param hop_length: int, length of the hopping window
    :return: array, normalized spectral centroid
    """
    spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    return normalize(spectral_cent)

# mfccs
def mfcc_comp(y, sr, n):
    """
    function that computes n MFCCs, its first, and second derivative with librosa
    :param y: audio file loaded with librosa
    :param sr: int, sample rate
    :param n: int, number of MFCCs to calculate
    :return: tuple with standard, first, and second order derivative mfccs, each contains a (n,feature length))-
    dimensional array with n entries per short-frame
    """
    mfccs = librosa.feature.mfcc(y=y, n_mfcc=n, sr=sr)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    return mfccs, delta_mfccs, delta2_mfccs

# lpccs
def autocorr(self, order=None):
    """
    code from https://www.kaggle.com/code/sourabhy/working-with-speech
    :param self:
    :param order:
    :return:
    """
    if order is None:
        order = len(self) - 1
    return [sum(self[n] * self[n + tau] for n in range(len(self) - tau)) for tau in range(order + 1)]

### sieht so bisschen aus wie absolut berechnet; in anderen Beispielen durch Länge und Varianz geteilt (normalisiert)


def core_lpcc(seq, err_term, order=None):
    """
    code mainly from https://www.kaggle.com/code/sourabhy/working-with-speech, slight changes
    :param seq:
    :param err_term:
    :param order:
    :return:
    """
    if order is None:
        order = len(seq) - 1
    lpcc_coeffs = [np.log(err_term), -seq[0]] # lpcc coeffs for i=0,1
    for n in range(2, order): # lpcc coeffs for i=2,...,order-1 of lpccs
        # Use order as upper bound for the last iteration (want 0,...,order-1 lpccs)
        upbound = (order if n > order - 1 else n)
        lpcc_coef = -sum(i * lpcc_coeffs[i] * seq[n - i - 1]
                         for i in range(1, upbound)) * 1. / upbound # sum over upper bound - 1
        lpcc_coef -= seq[n - 1] if n <= len(seq) else 0 # for both cases (m<p and m>p)
        lpcc_coeffs.append(lpcc_coef)
    return lpcc_coeffs


def lpcc(y, cepsorder):
    """
    code from https://www.kaggle.com/code/sourabhy/working-with-speech but changed compuation of error term
    :param lpcorder:
    :param cepsorder:
    :return:
    """
    coefs =  librosa.lpc(y, order=cepsorder)
    acseq =  np.array(autocorr(y, cepsorder))
    # err_term = np.sqrt(acseq[0] + sum(a * c for a, c in zip(acseq[1:], coefs)))
    b = np.hstack([[0], -1 * coefs[1:]])
    y_hat = scipy.signal.lfilter(b, [1], y)
    err_term = np.sum(np.square(y - y_hat)) # computation from librosa source code
    return core_lpcc(coefs, err_term, cepsorder)

def lowlevel_lpcc_comp(y, cepsorder):
    """
    computes LPCCs on a low level, i.e. LPCCs of order cepsorder for each frame
    :param y: audio sequence
    :param cepsorder: order of cepstrum
    :return:
    """
    frames = frames_gen(y)  # generate frames
    ll_lpccs = [lpcc(frame, cepsorder=cepsorder) for frame in frames]
    return np.array(ll_lpccs).transpose()

# lpcmfcc
def lpcmfcc_comp(lpccs, alpha=0.35, order_n=None):
    """
    computes LPCMFCCs based on the given LPCC coefficients recursively with order_n iterations going down
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
    computes LPCMFCCs on a low level, i.e. taking the computed LPCCs from each frame and computing LPCMFCCs for that
    frame
    :param lpccs_local: numpy array of shape (cepsorder, number of frames); all LPCCs per frame
    :param alpha: float; usually between 0.31 and 0.35 to be close to Mel scale
    :param order_n: int; number of iterations, generally equal to order of LPCCs
    :return: numpy array of shape (cepsorder, number of frames); all LPCMFCCs per frame
    """
    ll_lpcmfccs = [lpcmfcc_comp(lpccs_frame, alpha=alpha, order_n=order_n) for lpccs_frame in lpccs_local.transpose()]
    return np.array(ll_lpcmfccs).transpose()


def feature_extraction(filename, path):
    """

    :param filename:
    :param path:
    :return:
    """
    audio_path = os.path.join(path, filename)
    y, sr = librosa.load(audio_path, sr=None) # load audio data
    y_praat = parselmouth.Sound(audio_path)  # load praat sound

    label_dict = {'W':'anger', 'L':'boredom', 'E':'disgust', 'A':'fear', 'F':'happiness', 'T':'sadness', 'N':'neutral'}
    label = label_dict[filename[5]]
    speaker_num = filename[:2]
    gender_dict = {'03':'m', '08':'f', '09':'f', '10':'m', '11':'m', '12':'m', '13':'f', '14':'f', '15':'m', '16':'f'}
    gender = gender_dict[speaker_num]
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
    energy_avg_rate = average_change_rate(energy)
    duration_rising_energy, duration_falling_energy, value_rising_energy, value_falling_energy = rising_falling_slopes(energy)
    rms_energy = RMS_energy(y)
    log_rms = RMS_log_entropy(y)
    amplitude = amplitude_envelope(y)
    amplitude_avg_rate = average_change_rate(amplitude)
    lpc = lpc_est(y)

    # entropy computations
    spectral_ent = spectral_entropy(y, sr)
    shannon_ent = shannon_entropy(y)
    threshold_ent = threshold_entropy(y)
    log_energy_ent = log_energy_entropy(y)
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

    # mfccs, lpccs, lpcmfccs
    mfccs, delta_mfccs, delta2_mfccs = mfcc_comp(y, sr, n=13)
    lpccs_global = lpcc(y, cepsorder=12)
    lpcmfccs_global = lpcmfcc_comp(lpccs_global)
    lpccs_local = lowlevel_lpcc_comp(y, cepsorder=12)
    lpcmfccs_local = lowlevel_lpcmfcc_comp(lpccs_local)


    features_dict = {'file':filename, 'label':label, 'speaker':speaker_num, 'gender':gender, 'duration':duration,
                     'mean':mean, 'median':median, 'max':max, 'min':min, 'var':var, 'std':std, 'zcr':zcr,
                     'energy':energy, 'energy_avg_rate':energy_avg_rate, 'duration_rising_energy':duration_rising_energy,
                     'duration_falling_energy':duration_falling_energy, 'value_rising_energy':value_rising_energy,
                     'value_falling_energy':value_falling_energy, 'rms':rms_energy, 'log_rms':log_rms,
                     'amplitude':amplitude, 'amplitude_avg_rate':amplitude_avg_rate, 'lpc':lpc,
                     'spectral_entropy':spectral_ent, 'shannon_entropy':shannon_ent, 'threshold_entropy':threshold_ent,
                     'log_energy_entropy':log_energy_ent, 'sure_entropy':sure_ent, 'f0':f0, 'voiced':voiced_flag,
                     'f0_avg_change_rate':f0_avg_change_rate, 'pitch':pitch_values, 'pitch_time':pitch_time,
                     'duration_rising_pitch':duration_rising_pitch, 'duration_falling_pitch':duration_falling_pitch,
                     'value_rising_pitch':value_rising_pitch, 'value_falling_pitch':value_falling_pitch,
                     'speaking_rate':speaking_rate, 'articulation_rate':articulation_rate, 'asd':asd,
                     'spectral_centroid':spectral_cent, 'mfccs':mfccs, 'delta_mfccs':delta_mfccs,
                     'delta2_mfccs':delta2_mfccs, 'lpccs_local':lpccs_local, 'lpccs_global':lpccs_global,
                     'lpcmfccs_global':lpcmfccs_global, 'lpcmfccs_local':lpcmfccs_local} | formants

    return features_dict

def run_all_files(data_path, result_path):
    audio_files = os.listdir(data_path)

    # create an final list to store all results and convert to dataframe
    final_list = []

    i = 0
    with progressbar.ProgressBar(max_value=len(audio_files)) as bar:
        for file in audio_files:
            feature_dict = feature_extraction(file, data_path)
            final_list.append(feature_dict)
            i += 1
            bar.update(i)


    df = pd.DataFrame(final_list)
    df.to_csv(os.path.join(result_path, 'extracted_features.csv'), index=False)

    return df


def finalize_features(df):
    df['f0_max'] = df['f0'].apply(lambda x: np.max(x))
    df['f0_std'] = df['f0'].apply(lambda x: np.std(x))

    # calculate statistics for energy and pitch with rising/falling slopes
    statistics_features = ['energy', 'pitch']
    for feature in statistics_features:
        for index, row in df.iterrows():
            stats_list = []
            # create a list with maximum, mean, variance
            feature_row = row[feature].copy()
            stats_list.extend([np.max(feature_row), np.mean(feature_row), np.var(feature_row)]) ### in paper 4: they completely ignore 0 values -> should we do it for all?

            # calculate maximum, mean, median, and interquartile range of rising and falling slopes of duration and
            # value
            for elem in ['duration', 'value']:
                # clear lists from 0 values
                df[elem + '_rising_' + feature] = df[elem + '_rising_' + feature].apply(lambda x: x[x != 0])
                df[elem + '_falling_' + feature] = df[elem + '_falling_' + feature].apply(lambda x: x[x != 0])

                # max, mean, median, and iqr for rising slopes of feature
                feature_row = row[elem + '_rising_' + feature].copy()
                stats_list.extend([np.max(feature_row), np.mean(feature_row), np.median(feature_row),
                                   np.subtract(*np.percentile(feature_row, [75, 25]))])

                # max, mean, median, and iqr for falling slopes of feature
                feature_row = row[elem + '_falling_' + feature].copy()
                stats_list.extend([np.max(feature_row), np.mean(feature_row), np.median(feature_row),
                                   np.subtract(*np.percentile(feature_row, [75, 25]))])

                df.loc[[index], feature + '_stats'] = pd.Series([stats_list], index=df.index[[index]])

            df.drop(columns=[elem + '_rising_' + feature, elem + '_falling_' + feature], inplace=True)

    # further stats for pitch, energy
    df['pitch_values_non0'] = df['pitch_values'].apply(lambda x: x[x != 0])
    df['skew_log_pitch'] = df['pitch_values_non0'].apply(lambda x: skew(np.log(x)))
    df['range_log_pitch'] = df['pitch_values_non0'].apply(lambda x: np.abs((np.max(x) - np.min(x))))
    df['energy_non0'] = df['energy'].apply(lambda x: x[x != 0])
    df['range_log_energy'] = df['energy_non0'].apply(lambda x: np.abs((np.max(x) - np.min(x))))

    # calculate statistics of cepstrum coefficients mfccs, lpccs
    cepstrum_coeffs = ['mfccs', 'lpccs_local', 'lpcmfccs_local']
    for feature in cepstrum_coeffs:
        for index, row in df.iterrows():
            stats_list = []
            for i, coef in enumerate(row[feature]):
                # create a list with mean, variance, maximum, and minumum of all MFCCs/LPCCs/LPCMFCCs respectively
                stats_list.extend([np.mean(coef), np.var(coef), np.max(coef), np.min(coef)])
            df.loc[[index], feature + '_stats'] = pd.Series([stats_list], index=df.index[[index]])


    df.to_csv(os.path.join(result_path, 'extracted_features_modified.csv'), index=False)

    return df


"""
if __name__ == "__main__":
    #lang = getoptions()
    lang = 'de'
    l = lexiconGenerator(lang)
    l.runLexiconGeneration()
"""


#dict1 = feature_extraction('03a01Fa.wav', data_path)
dict2 = run_all_files(data_path, result_path)



audio1_path = os.path.join(data_path, "03a01Fa.wav")
audio1, sample_rate1 = librosa.load(audio1_path, sr=None)

test = 0