
import os 
import glob
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split


def split_train_val_test(filenames, labels, seed):
  # train test split
  train_files, raw_test_files, train_labels, raw_test_labels = train_test_split(
    filenames, labels, test_size=0.30, random_state=seed, 
    shuffle=True, stratify=labels) 
  
  # split the test set in two halfs, the first for validation and the second for testing 
  half_test = len(raw_test_files) // 2
  val_files, val_labels = raw_test_files[:half_test], raw_test_labels[:half_test, :]
  test_files, test_labels = raw_test_files[half_test:], raw_test_labels[half_test:, :]
  
  unique_labels, files_by_labels = group_files_by_labels(train_files, train_labels)

  sets = {
    'train': {'files': train_files, 'labels': train_labels, 
      'unique_labels': unique_labels, 'files_by_labels': files_by_labels},
    'test': {'files': test_files, 'labels': test_labels},
    'val': {'files': val_files, 'labels': val_labels},
  }

  return sets


def group_files_by_labels(train_files, train_labels):
  unique_labels = np.unique(train_labels, axis=0)
  files_by_labels = []
  for label in unique_labels:
    files_of_label = []
    for i in range(len(train_files)):
      f = train_files[i]
      l = train_labels[i]
      if (l == label).all():
        files_of_label.append(f)
    files_by_labels.append(files_of_label)
  return unique_labels, files_by_labels


def get_label(file_path):
  parts = file_path.split(os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this 
  # to work in a TensorFlow graph.
  return parts[-2]


def get_wav_files_and_labels(path_ds):
  if not os.path.exists(path_ds):
    msg = f"Path {path_ds} not found."
    raise Exception(msg)
  data_dir = pathlib.Path(path_ds)
  filenames = glob.glob(str(data_dir) + '/*/*wav')
  labels = [get_label(f) for f in filenames]
  #print(labels)
  filenames = np.array(filenames)
  labels = np.array(labels)
  return filenames, labels 

