
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'  # turn off tensorflow verbosity
import tensorflow as tf
from bac.preproc import *
from bac.inout import *
from bac.models import *
from bac.traineval import *
from bac.dataset import *


# ---------------------------------------------------------
# Only required when running in Ifes' computers.
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession

#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)
# --------------------------------------------------------


def configure_environment(seed):
  tf.random.set_seed(seed)
  np.random.seed(seed)
  tf.get_logger().setLevel('ERROR')
  tf.autograph.set_verbosity(3)


def main():
  ##########################################
  # basic parameters 
  ##########################################
  DEBUG = False
  #MAX_WAVE_SIZE = 905021    # true maximum size
  MAX_WAVE_SIZE = 135000     # subsampled size
  SEED = 42
  MODEL = 'wagner_lstm_conv1d_1'    # a lista completa de modelos esta' em bac.models.py:get_model() 
  NUM_EPOCHS = 250 
  BATCH_SIZE = 16
  LEARNING_RATE = 1e-4
  BALANCE_BATCHES = False
  #PATH_DS = /content/drive/MyDrive/BraccentTensor/Mono/
  #PATH_DS = 'C:\\Users\\filip\\Desktop\\Filipe\\projetos\\2020-wagner-transcricao\\BRAccent\\Mono\\'
  PATH_DS = 'C:/AiShellAccent'
  EXPERIMENT_NAME = f"experiment_{time.time()}_{MODEL}".replace(".", "_")

  configure_environment(SEED)

  ##########################################
  # prepare and load dataset
  ##########################################
  filenames, labels = get_wav_files_and_labels(PATH_DS)
  filenames, labels = shuffle_data(filenames, labels)
  labels, encoder = convert_labels_to_onehot(labels, debug=DEBUG)
  sets = split_train_val_test(filenames, labels, SEED)

  ##########################################
  # build model, loss function and optimizer
  ##########################################
  input_shape, preprocessing_fn = get_model_config(MODEL, sets['train']['files'][0], MAX_WAVE_SIZE)
  model = get_model(MODEL, input_shape, len(encoder.categories_[0]))
  model.summary()
  loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)

  ##########################################
  # train and then evaluate the model with the test set
  ##########################################
  experiment_dir = create_experiment_directory(EXPERIMENT_NAME)
  #checkpoint_name = f"{experiment_dir}/{EXPERIMENT_NAME}_model.ckpt"
  checkpoint_name = f"results/{EXPERIMENT_NAME}_model.ckpt"
  train_history = train(model, loss_fn, optimizer, sets['train'], sets['val'], 
    preprocessing_fn, NUM_EPOCHS, BATCH_SIZE, checkpoint_name, BALANCE_BATCHES)
  model = tf.keras.models.load_model(checkpoint_name)
  test_history = evaluate_model(model, preprocessing_fn, sets['test'], BATCH_SIZE)
  
  ##########################################
  # print and save results
  ##########################################
  print("\nResults on Test set:")
  print_evaluation_report(test_history, encoder.categories_[0])
  print("Saving results and reports.")
  save_results_and_reports(experiment_dir, sets, train_history, test_history, encoder.categories_[0])

  print("OK.")
  return


if __name__ == "__main__":
  main()

