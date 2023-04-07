# import packages
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import librosa
import antropy as ant # for entropy computation
import scipy
from scipy.stats import entropy
import parselmouth
from parselmouth import praat

data_path = os.path.join(str(Path(__file__).parents[1]), 'data/wav')

def frames_gen(y, center=True, frame_length=2048, hop_length=512, pad_mode="constant"):
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
    return sklearn.preprocessing.MinMaxScaler().fit_transform(np.array(x).reshape(-1,1))

def energy_comp(y):
    """

    :param y:
    :return:
    """
    frames = frames_gen(y) # generate frames
    ener = [np.sum(np.square(frame)) for frame in frames]
    return normalize(ener)

def RMS_energy(y, frame_length=2048, hop_length=512):
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

def amplitude_envelope(y, frame_length=2048, hop_length=512):
    """

    :param y:
    :param frame_length:
    :param hop_length:
    :return:
    """
    return np.array([max(y[i:i+frame_length]) for i in range(0, y.size, hop_length)])

def ZCR(y, frame_length=2048, hop_length=512):
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
    return spectral

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
    return [np.mean(frame) for frame in frames]

def log_energy_entropy(y):
    """

    :param y:
    :return:
    """
    frames = frames_gen(y)

    # infinity is replaced by the largest finite floating point values representable
    log_entropy = np.nan_to_num([np.sum(np.log(np.square(frame))) for frame in frames])
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

    :param y:
    :param sr:
    :return:
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), hop_length=512)
    f0 = np.nan_to_num(f0) # convert nan values to 0
    return f0, voiced_flag, voiced_prob

def formant_analysis(filename, path, gender, formant_order=4, f0min = 75, f0max = 600):
    """
    in the example it was 300; but here we see standard is 600
    (https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__ac____.html)
    :param filename:
    :param path:
    :param gender:
    :param formant_order:
    :param f0min:
    :param f0max:
    :return:
    """
    y = parselmouth.Sound(os.path.join(path, filename))  # praat sound

    # compute the occurrences of periodic instances in the signal
    pointProcess = praat.call(y, "To PointProcess (periodic, cc)", f0min, f0max)

    # define maximal frequency depending on gender
    if gender == 'female':
        formant_ceiling = 5500
    else:
        formant_ceiling = 5000

    formants = praat.call(y, "To Formant (burg)", 0.0025, 5, formant_ceiling, 0.025, 50)  # formants definition

    # assign formant values with times where they make sense (periodic instances)
    numPoints = praat.call(pointProcess, "Get number of points")

    # define a dictionary to save the formants of the specified order and initialize with empty list
    form_dict = {}
    for order in range(1, formant_order + 1):
        form_dict['f' + str(order) + '_median'] = []

        for point in range(0, numPoints):
            point += 1
            t = praat.call(pointProcess, "Get time from index", point)
            formant = praat.call(formants, "Get value at time", order, t, 'Hertz', 'Linear')
            if str(formant) != 'nan':
                form_dict['f' + str(order) + '_median'].append(formant)
        form_dict['f' + str(order) + '_median'] = np.median(form_dict['f' + str(order) + '_median'])

    return form_dict


def feature_extraction(filename, path):
    """

    :param filename:
    :param path:
    :return:
    """
    audio_path = os.path.join(path, filename)
    y, sr = librosa.load(audio_path, sr=None) # load audio data

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
    rms_energy = RMS_energy(y)
    log_rms = RMS_log_entropy(y)
    amplitude = amplitude_envelope(y)
    lpc = lpc_est(y)

    # entropy computations
    spectral_ent = spectral_entropy(y, sr)
    shannon_ent = shannon_entropy(y)
    threshold_ent = threshold_entropy(y)
    log_energy_ent = log_energy_entropy(y)
    sure_ent = sure_entropy(y)

    # f0, formants
    f0, voiced_flag, voiced_prob = f0_comp(y, sr)
    formants = formant_analysis(filename, path, gender)

    features_dict = {'file':filename, 'label':label, 'speaker':speaker_num, 'gender':gender, 'duration':duration,
                     'mean':mean, 'median':median, 'max':max, 'min':min, 'var':var, 'std':std, 'zcr':zcr,
                     'energy':energy, 'rms':rms_energy, 'log_rms':log_rms, 'amplitude':amplitude, 'lpc':lpc,
                     'spectral_entropy':spectral_ent, 'shannon_entropy':shannon_ent, 'threshold_entropy':threshold_ent,
                     'log_energy_entropy':log_energy_ent, 'sure_entropy':sure_ent, 'f0':f0, 'voiced':voiced_flag} | formants

    return features_dict

"""
if __name__ == "__main__":
    #lang = getoptions()
    lang = 'de'
    l = lexiconGenerator(lang)
    l.runLexiconGeneration()
"""

dict1 = feature_extraction('03a01Fa.wav', data_path)
test = 0