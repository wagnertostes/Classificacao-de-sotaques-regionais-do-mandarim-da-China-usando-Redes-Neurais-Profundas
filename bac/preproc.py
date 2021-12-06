import os
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_io as tfio
from sklearn.preprocessing import OneHotEncoder
from bac.inout import *

def convert_labels_to_onehot(labels, debug=False):
  # OneHotEncoder expects 2-dim data with number of rows equals to 
  # the number of samples and number of columns equals to the class dimension.
  labels = labels.reshape((len(labels), 1))
  
  #print(labels)
  enc = OneHotEncoder(sparse=False)
  labels_onehot = enc.fit_transform(labels)

  if debug:
    print_debug_info_about_labels(labels, labels_onehot, enc)

  return labels_onehot, enc


def shuffle_data(filenames, labels):
  indices = np.arange(len(filenames))
  np.random.shuffle(indices)
  filenames = filenames[indices]
  labels = labels[indices]
  return filenames, labels


def load_and_preprocess_waveform(file_path, max_size):
  audio_binary = tf.io.read_file(file_path)

  '''
    The audio file will initially be read as a binary file, which you'll want to convert into a numerical tensor.

    To load an audio file, you will use tf.audio.decode_wav, which returns the WAV-encoded audio as a Tensor and the sample rate.

    A WAV file contains time series data with a set number of samples per second. Each sample represents the amplitude of the audio 
    signal at that specific time. In a 16-bit system, like the files in mini_speech_commands, the values range from -32768 to 32767. 
    The sample rate for this dataset is 16kHz. Note that tf.audio.decode_wav will normalize the values to the range [-1.0, 1.0].
  '''
  waveform, _ = tf.audio.decode_wav(audio_binary)
  #waveform =  tfio.audio.decode_mp3(audio_binary)
  waveform = tf.squeeze(waveform, axis=-1)
  waveform = tf.cast(waveform, tf.float32)

  # TODO: trim 
  # eps = np.max(np.abs(waveform)) * 0.01
  # position = tfio.experimental.audio.trim(waveform, axis=0, epsilon=eps)
  # start = position[0]
  # end = position[1]
  # print("start, end:", start, end)
  # waveform = waveform[start:end]

  # zero padding
  if tf.shape(waveform) < max_size:
    zero_padding = tf.zeros(max_size - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)

  # reduce the waveform size if necessary
  waveform = waveform[:max_size]
  
  return waveform.numpy()


def compute_spectogram(waveform):
    ## ***********************
    ## Typical ouput is 97x2049 representing 97 timesteps and 2049 frequencies.
    ## ***********************
    spectrogram = tf.signal.stft(waveform, frame_length=3000, frame_step=2000)
    spectrogram = tf.abs(spectrogram)
    return spectrogram.numpy()


def compute_mel_spectrogram(waveform):
    mel_spectrogram = tfio.experimental.audio.melscale(waveform, rate=16000, mels=128, fmin=0, fmax=8000)  
    dbscale_mel_spectrogram = tfio.experimental.audio.dbscale(mel_spectrogram, top_db=80)
    freq_mask = tfio.experimental.audio.freq_mask(dbscale_mel_spectrogram, param=10)
    return freq_mask


def get_model_config(model, sample, max_wave_size):
    # function to load a waveform from a file
    wave_preprocessing_fn = lambda f: load_and_preprocess_waveform(f, max_wave_size) 
    # function to load a waveform and then compute the spectrogram
    spectro_preprocessing_fn = lambda f: compute_spectogram(load_and_preprocess_waveform(f, max_wave_size))

    # get the input shape
    wave_shape = wave_preprocessing_fn(sample).shape
    spectro_shape = spectro_preprocessing_fn(sample).shape

    if model == 'cnn1d':
        return wave_shape, wave_preprocessing_fn
    elif 'cnn2d' in model:
        return spectro_shape, spectro_preprocessing_fn
    elif model == 'wagner_lstm' or model == 'wagner_bilstm':
        return spectro_shape, spectro_preprocessing_fn
    elif model == 'wagner_lstm_conv1d_1' or model == 'wagner_bilstm_conv1d_1':
        return spectro_shape, spectro_preprocessing_fn
    elif model == 'wagner_lstm_conv1d_2' or model == 'wagner_bilstm_conv1d_2':
        return spectro_shape, spectro_preprocessing_fn
    elif model == 'wagner_lstm_conv1d_3' or model == 'wagner_bilstm_conv1d_3':
        return spectro_shape, spectro_preprocessing_fn
    else:
        raise Exception(f"Model {model} not found.")
