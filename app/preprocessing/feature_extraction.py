import librosa
from preprocessing.wav_handler import WavHandler
import tensorflow as tf
import pandas as pd

def extract_features_from_file(file_path):
  dict_features = {}
  train_time_slice = 1
  file_duration = librosa.get_duration(path=file_path)

  for i in range(0, int(file_duration), train_time_slice):
    sliced_part_file_path = file_path.replace(".wav", f"#{i}.wav")
    wav_handler = WavHandler(file_path, float(i), duration=float(train_time_slice))
    dict_features[sliced_part_file_path] = wav_handler.get_features()

  return dict_features

def convert_to_dataframe(dict_features):
  features = pd.DataFrame.from_dict(dict_features, orient='index',
                                    columns=['harmonic', 'mfcc', 'spectro', 'chroma', 'contrast'])

  mfcc = pd.DataFrame(features.mfcc.values.tolist(), index=features.index)
  mfcc = mfcc.add_prefix('mfcc_')

  spectro = pd.DataFrame(features.spectro.values.tolist(),index=features.index)
  spectro = spectro.add_prefix('spectro_')

  chroma = pd.DataFrame(features.chroma.values.tolist(),index=features.index)
  chroma = chroma.add_prefix('chroma_')

  contrast = pd.DataFrame(features.contrast.values.tolist(),index=features.index)
  #this is by purpose: it is done like this in the preprocessing in the notebook
  #FIX THIS!!!!!
  contrast = chroma.add_prefix('contrast_')

  features = features.drop(labels=['mfcc', 'spectro', 'chroma', 'contrast'], axis=1)
  df_features = pd.concat([features, mfcc, spectro, chroma, contrast], axis=1, join='inner')

  targets = []
  for name in df_features.index.tolist():
    targets.append(name)
  df_features['targets'] = targets
  df_features = df_features.drop(labels=['targets'], axis=1)
  df_features.info()
  return df_features


def prepare_for_model(df_features):
  features = df_features.astype(float)
  features = tf.convert_to_tensor(features)
  features = tf.expand_dims(tf.expand_dims(features, axis=-1), axis=-1)

  return features

def get_audio_data(file_path):
  dict_features = extract_features_from_file(file_path)
  df_features = convert_to_dataframe(dict_features)

  return prepare_for_model(df_features)