
import time 
import numpy as np 
import tensorflow as tf 
from bac.inout import * 


def get_num_batches_per_epoch(files, batch_size):
  n_files = len(files)
  n_batches_per_epoch = n_files // batch_size
  if len(files) % batch_size != 0:
    n_batches_per_epoch += 1
  return n_batches_per_epoch


def sample_batch(data, preprocessing_fn, batch_id, batch_size):
  begin = batch_id * batch_size 
  end = (batch_id + 1) * batch_size 
  files = data['files'][begin:end]
  labels = data['labels'][begin:end, :]
  input_data = [preprocessing_fn(f) for f in files]
  return np.array(input_data), labels


def sample_balanced_batch(data, preprocessing_fn, batch_id, batch_size):
  batch_x, batch_y = [], []
  for i in range(batch_size):
    lbl_idx = np.random.randint(len(data['unique_labels']))
    label = data['unique_labels'][lbl_idx]
    sample_idx = np.random.randint(len(data['files_by_labels'][lbl_idx]))
    sample = data['files_by_labels'][lbl_idx][sample_idx]
    batch_x.append(preprocessing_fn(sample))
    batch_y.append(label)
  return np.array(batch_x), np.array(batch_y)
  

def update_train_history(history, batch_y, prediction, loss):
  n_hits = np.sum(np.argmax(batch_y, axis=-1) == np.argmax(prediction, axis=-1))
  n_samples = len(batch_y)
  history['n_hits'].append(n_hits)
  history['n_samples'].append(n_samples)
  history['accuracy'].append(n_hits / n_samples)
  history['loss'].append(loss)


@tf.function
def train_batch(model, model_vars, loss_fn, optimizer, batch_x, batch_y):
  with tf.GradientTape() as tape:
    prediction = model(batch_x, training=True)
    loss = loss_fn(batch_y, prediction)
  grads = tape.gradient(loss, model_vars)
  optimizer.apply_gradients(zip(grads, model_vars))
  return prediction, loss


def train_epoch(model, model_vars, preprocessing_fn, loss_fn, optimizer, 
  train_data, n_train_batches, batch_size, balance_batches):
  history = {'loss': [], 'accuracy': [], 'n_hits': [], 'n_samples': []}
  progBar = tf.keras.utils.Progbar(len(train_data['files']), stateful_metrics=['loss', 'train_acc'])

  for batch_id in range(n_train_batches):
    if balance_batches:
      batch_x, batch_y = sample_balanced_batch(train_data, preprocessing_fn, batch_id, batch_size)
    else:
      batch_x, batch_y = sample_batch(train_data, preprocessing_fn, batch_id, batch_size)
    prediction, loss = train_batch(model, model_vars, loss_fn, optimizer, batch_x, batch_y)
    update_train_history(history, batch_y, prediction, loss)
    progBar.update(batch_id * batch_size, values=[('loss', history['loss'][-1]), ('train_acc', history['accuracy'][-1])]) 

  return history


def update_eval_history(history, batch_y, prediction):
  predictions = np.argmax(prediction, axis=-1)
  groundtruth = np.argmax(batch_y, axis=-1)
  n_hits = np.sum(groundtruth == predictions)
  loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  loss_eval = loss_fn(batch_y, prediction)
  n_samples = len(batch_y)
  history['n_hits'].append(n_hits)
  history['n_samples'].append(n_samples)
  history['accuracy'].append(n_hits / n_samples)
  history['loss'].append(loss_eval)
  history['predictions'].extend(predictions)
  history['groundtruth'].extend(groundtruth)


def evaluate_model(model, preprocessing_fn, data, batch_size):
  history = {'accuracy': [], 'loss': [], 'n_hits': [], 'n_samples': [], 'predictions': [], 'groundtruth': []}
  
  n_batches = get_num_batches_per_epoch(data['files'], batch_size)
  for batch_id in range(n_batches):
    batch_x, batch_y = sample_batch(data, preprocessing_fn, batch_id, batch_size)
    prediction = model(batch_x, training=False)
    update_eval_history(history, batch_y, prediction)  
  
  return history


def update_history(history, epoch_id, train_history, val_history):
  mean_train_loss_on_batches = np.mean(train_history['loss'])
  sum_train_hits_on_batches = np.sum(train_history['n_hits'])
  num_trained_samples = np.sum(train_history['n_samples'])
  train_accuracy = sum_train_hits_on_batches / num_trained_samples

  sum_val_hits_on_batches = np.sum(val_history['n_hits']) 
  num_val_samples = np.sum(val_history['n_samples'])
  val_accuracy = sum_val_hits_on_batches / num_val_samples

  history['train']['loss'].append(mean_train_loss_on_batches)
  history['train']['accuracy'].append(train_accuracy)
  history['val']['accuracy'].append(val_accuracy)

  history['detailed_train']['loss'].extend(train_history['loss'])
  history['detailed_train']['accuracy'].extend(train_history['accuracy'])
  history['detailed_val']['accuracy'].extend(val_history['accuracy'])

  if val_accuracy > history['max_val']:
    history['max_val'] = val_accuracy
    history['max_val_epoch'] = epoch_id


def train(model, loss_fn, optimizer, train_data, val_data, preprocessing_fn, 
  num_epochs, batch_size, checkpoint_name, balance_batches):
  training_start_time = time.time()
  n_train_batches = get_num_batches_per_epoch(train_data['files'], batch_size)

  model_vars = model.trainable_variables
  history = {
    'train': {'loss': [], 'accuracy': []}, 
    'val': {'accuracy': []}, 
    'detailed_train': {'loss': [], 'accuracy': []}, 
    'detailed_val': {'accuracy': []}, 
    'max_val': -np.inf,
    'max_val_epoch': -1,
  }

  training_times = []

  for epoch_id in range(num_epochs):
    epoch_start = time.time()
    print("\nEpoch %d / %d" % (epoch_id, num_epochs))

    train_history = train_epoch(model, model_vars, preprocessing_fn, loss_fn, 
      optimizer, train_data, n_train_batches, batch_size, balance_batches)
    
    val_history = evaluate_model(model, preprocessing_fn, val_data, batch_size)
    
    update_history(history, epoch_id, train_history, val_history)

    # best validation accuracy was achieved in this epoch
    if history['max_val_epoch'] == epoch_id:
      model.save(checkpoint_name)

    print_epoch_report(epoch_id, history)
    print_evaluation_report(val_history)
    print_time_report(epoch_start, training_start_time, 
      training_times, epoch_id, num_epochs)
    
  return history


