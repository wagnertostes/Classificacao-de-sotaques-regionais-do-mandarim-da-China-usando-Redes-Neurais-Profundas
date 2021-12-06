import os 
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
plt.switch_backend('Agg')


def plot_sample_spectrograms(sets, preprocessing_fn):
  rows = 3
  cols = 5

  _, axes = plt.subplots(rows, cols, figsize=(10, 12))

  for row, ds in enumerate(['train', 'val', 'test']):
    for col in range(cols):
      spectrogram = preprocessing_fn(sets[ds]['files'][col])
      ax = axes[row][col]
      plot_spectrogram(spectrogram, ax)
      ax.set_title(ds + " " + str(sets[ds]['labels'][col]))

  plt.show()


def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  spectrogram = np.transpose(spectrogram)
  log_spec = np.log(spectrogram)
  height = log_spec.shape[0]
  X = np.arange(log_spec.shape[1]) 
  Y = np.arange(height)
  ax.pcolormesh(X, Y, log_spec)


def plot_sample_waveforms(sets, preprocessing_fn):
  rows = 3
  cols = 5

  _, axes = plt.subplots(rows, cols, figsize=(10, 12))

  for row, ds in enumerate(['train', 'val', 'test']):
    for col in range(cols):
      audio = preprocessing_fn(sets[ds]['files'][col])
      ax = axes[row][col]
      ax.plot(audio)
      ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
      ax.set_title(ds + " " + str(sets[ds]['labels'][col]))
  
  plt.show()


def print_time_report(epoch_start, training_start_time, training_times, epoch_id, num_epochs):
    now = time.time()
    training_times.append(now - epoch_start)
    avg_train_time = np.mean(training_times)
    print("total-training-time: %.2fmin avg-epoch-train: %.4fs ETA: %.2fmin" % 
      ((now - training_start_time) / 60., avg_train_time, ((num_epochs - epoch_id - 1) * avg_train_time) / 60.))


def print_epoch_report(epoch_id, history):
    msg = "\nEpoch %d - train loss: %.4f train acc: %.3f val acc: %.3f max val acc: %.3f"
    msg = msg % (epoch_id, history['train']['loss'][-1], history['train']['accuracy'][-1], 
      history['val']['accuracy'][-1], np.max(history['val']['accuracy']))
    print(msg)


def print_evaluation_report(history, labels_names=None):
    print("\nConfusion Matrix:")
    print(confusion_matrix(history['groundtruth'], history['predictions']))
    print("\nClassification Report:")
    #print(classification_report(history['groundtruth'], history['predictions'], target_names=labels_names, zero_division=0))
    print(classification_report(history['groundtruth'], history['predictions'], target_names=labels_names))
    print("\nNumber of correctly classified samples: %d of %d" % (np.sum(history['n_hits']), len(history['groundtruth'])))
    print("Accuracy (just for double check): %.4f" % (np.sum(history['n_hits']) / len(history['groundtruth'])))
    print("Loss                            : %.4f" % (np.min(history['loss'])))


def create_experiment_directory(experiment_name):
  if not os.path.exists("results"):
    os.mkdir("results")
  experiment_dir = f"results/{experiment_name}"
  if not os.path.exists(experiment_dir):
    os.mkdir(experiment_dir)
  return experiment_dir


def save_results_and_reports(experiment_dir, sets, train_history, test_history, labels_names):
 if len(train_history) > 0:
  save_dataset(f"{experiment_dir}/train_set.csv", sets['train'])
  save_dataset(f"{experiment_dir}/val_set.csv", sets['val'])
  save_dataset(f"{experiment_dir}/test_set.csv", sets['test'])

  train_history_epoch_name = f"{experiment_dir}/train_history_by_epoch.csv"
  save_train_history(train_history_epoch_name, 
    train_history['train']['loss'], 
    train_history['train']['accuracy'], 
    train_history['val']['accuracy'])

  train_history_batch_name = f"{experiment_dir}/train_history_by_batch.csv"
  save_train_history(train_history_batch_name, 
    train_history['detailed_train']['loss'], 
    train_history['detailed_train']['accuracy'])

  accuracy_by_epoch_name = f"{experiment_dir}/accuracy_by_epoch.png"
  plot_and_save_accuracy_evolution(accuracy_by_epoch_name, 
    train_history['train']['accuracy'], 
    train_history['val']['accuracy'])

 test_predictions_name = f"{experiment_dir}/test_predictions.csv"
 test_metrics_name = f"{experiment_dir}/test_metrics.txt"
 save_test_predictions(test_predictions_name, test_history)
 save_test_results(test_metrics_name, test_history, labels_names)


def save_dataset(filename, dataset):
  with open(filename, "w") as f:
    f.write("file;label\n")
    for fl, l in zip(dataset['files'], dataset['labels']):
      label_as_int = np.argmax(l, axis=-1)
      f.write(f"{fl};{label_as_int}\n")


def save_train_history(filename, train_loss, train_acc, val_acc=None):
  with open(filename, "w") as f:
    f.write("id;train_loss;train_acc;val_acc;val_loss\n")
    for i in range(len(train_loss)):
      line = "%d;%f;%f;%f;\n"
      val = -1 if val_acc is None else val_acc[i]
      line = line % (i, train_loss[i], train_acc[i], val)
      f.write(line)


def save_test_predictions(filename, history):
  with open(filename, "w") as f:
    f.write("gt;prediction\n")
    for gt, prd in zip(history['groundtruth'], history['predictions']):
      f.write(f"{gt};{prd}\n")


def save_test_results(filename, history, labels_names):
  with open(filename, "w") as f:
    f.write("Classification Report:\n")
    f.write(classification_report(history['groundtruth'], history['predictions'], 
      target_names=labels_names))
    f.write("Los:\n")
    f.write(str(np.min(history['loss'])))
    f.write("\nConfusion Matrix:\n")
    cm = confusion_matrix(history['groundtruth'], history['predictions'])
    for row in cm:
      for value in row:
        f.write(f"{value}\t")
      f.write("\n")


def plot_and_save_accuracy_evolution(filename, train_acc, val_acc):
  sns.set_theme()
  epochs = np.arange(len(train_acc)) + 1
  plt.legend(['train', 'validation'])
  plt.ylim([-0.05, 1.05])
  plt.ylabel("Accuracy")
  plt.xlabel("Epochs")
  plt.yticks(np.arange(0.0, 1.05, 0.1))
  plt.xticks(epochs)
  plt.plot(epochs, train_acc)
  plt.plot(epochs, val_acc)
  plt.savefig(filename)

