"""
train_model.py
This script provides a training pipeline for PyTorch-based classification and segmentation models.
It supports multiple model architectures (VGG8, UnetS) and configurable hyperparameters for hyperparameter optimization.
The script handles dataset loading, model selection, training, logging, and plotting of loss curves.
Device-agnostic code ensures compatibility with both CPU and CUDA-enabled GPUs.
Part of this code has been implemented from: https://www.learnpytorch.io/05_pytorch_going_modular/

-----------------------------
Hyperparameters:
- TRANSFORM (bool): Wheter to apply transformations to the dataset.
- AMOUNT_SAMPLES (int): Number of samples to generate for each gas source location during augmentation.
- WINDOW_SIZE (list of int): Size of the window to use for data augmentation.
- NUM_EPOCHS (int): Number of epochs to train the model.
- BATCH_SIZE (int): Batch size for training.
- LEARNING_RATE (float): Learning rate for the optimizer.

Training Parameters:
- MODEL_TYPES (list of str): Which models are available for training. ("VGG-8", "U-NetS")
- TRAIN_MODEL (int): Index of the model to train (0 for VGG-8, 1 for U-NetS).
- LOGS_SAVE (bool): Whether to save or show logs during dataset initialization and processing.
- SAVE_DATASET (bool): Whether to save the processed dataset.
- RESET_TRAINING (bool): Whether to reset training to start from scratch (deletes models).
- LOAD_SEED (int): Random seed for loading the dataset.
- TRAIN_SEED (int): Random seed for training.
- TEST_SEED (int): Random seed for testing.

Testing Hyperparameters:
- HYPERPARAMETER_OPTIMIZATION (bool): Whether to perform hyperparameter optimization.
- BATCH_SIZE (list of int): List of batch sizes to test.
- LEARNING_RATE (list of float): List of learning rates to test.
- AMOUNT_SAMPLES (int): Number of samples to generate for each gas source location during augmentation.
- WINDOW_SIZE (list of int): Size of the window to use for data augmentation.

Constants:
- device (str): Target device for training (CPU or CUDA). [Training has been performed on GPU -> 9x times faster than CPU training]]

-----------------------------
Functions:
- main():
  Initializes logging and starts the training process for the selected model.

- train_all_models(model_type, transform):
  Trains the specified model type with the given transformations.
  Sets up the dataloader, model, optimizer, and loss function.
  Handles training and testing of the model, logging results and plotting loss curves.

-----------------------------    
Dependencies:
- torch, logging, timeit
- Custom modules: logs.logger, model_dataloader, utils, engine, model_builder, engine_encdec

-----------------------------
Usage:
- Run this script directly to initialize and start the training process:
    python train_model.py
"""


import torch
import logging
from logs import logger
import model_dataloader, utils, engine, model_builder,engine_encdec
from timeit import default_timer as timer 

# Setup hyperparameters
HYPER_PARAMETERS = {
              "TRANSFORM": True,
              "AMOUNT_SAMPLES": 16,
              "WINDOW_SIZE": [64,64],
              "NUM_EPOCHS": 10,
              "BATCH_SIZE": 32,
              "LEARNING_RATE": 0.001,               
  }


TRAINING_PARAMETERS = {
               "MODEL_TYPES": ["VGG8", "UnetS"],
               "TRAIN_MODEL": 1,         # 0 for VGG8, 1 for UnetS
               "LOGS_SAVE": False,       # Save logs to file or show in console, when training preferred to set to True (better overview and additional logs for later analysis)
               "SAVE_DATASET": False,
               "RESET_TRAINING": False,  # Reset training to start from scratch
               "LOAD_SEED": 16923,
               "TRAIN_SEED": 42,
               "TEST_SEED": 1009
  }


TESTING_HYPERPARAMETERS = {
              "HYPERPARAMETER_OPTIMIZATION": True,
              "BATCH_SIZE" : [16, 32, 64, 128, 256],
              "LEARNING_RATE": [0.001, 0.01, 0.1, 1.0],
              "AMOUNT_SAMPLES": 1,
              "WINDOW_SIZE": [64, 64],
  }

if TESTING_HYPERPARAMETERS["HYPERPARAMETER_OPTIMIZATION"]:
  HYPER_PARAMETERS['LEARNING_RATE'] = TESTING_HYPERPARAMETERS["LEARNING_RATE"][0]
  HYPER_PARAMETERS['BATCH_SIZE'] = TESTING_HYPERPARAMETERS["BATCH_SIZE"][2]
  HYPER_PARAMETERS['AMOUNT_SAMPLES'] = TESTING_HYPERPARAMETERS["AMOUNT_SAMPLES"]
  HYPER_PARAMETERS['WINDOW_SIZE'] = TESTING_HYPERPARAMETERS["WINDOW_SIZE"]

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
  model_type=TRAINING_PARAMETERS['MODEL_TYPES'][TRAINING_PARAMETERS["TRAIN_MODEL"]]
  transform=HYPER_PARAMETERS['TRANSFORM']
  logs_save=TRAINING_PARAMETERS['LOGS_SAVE']
  logger.logging_config(logs_save=logs_save,amount_samples=HYPER_PARAMETERS['AMOUNT_SAMPLES'], transform=transform, model_type=model_type, window_size=HYPER_PARAMETERS['WINDOW_SIZE'])
  if TRAINING_PARAMETERS['RESET_TRAINING']:
    utils.reset_training(model_type=model_type, transform=transform)  
  train_all_models(model_type=model_type, transform=transform)
   




    

def train_all_models(model_type= "VGG", transform=True):
  #Set Random Load Seed
  utils.seed_generator(SEED=TRAINING_PARAMETERS['LOAD_SEED'])

  # Create dataloader and model
  train_dataloader,test_dataloader,classes = model_dataloader.create_dataloader(model_type=model_type, batch_size=HYPER_PARAMETERS['BATCH_SIZE'], transform=transform, amount_samples=HYPER_PARAMETERS['AMOUNT_SAMPLES'], window_size=HYPER_PARAMETERS['WINDOW_SIZE'])
  model = model_builder.choose_model(model_type=model_type,output_shape=classes,device=device, window_size=HYPER_PARAMETERS['WINDOW_SIZE'])

  # Set optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=HYPER_PARAMETERS['LEARNING_RATE'])

  # Set loss function for VGG-8
  loss_fn = torch.nn.CrossEntropyLoss()     
  
  #Training + Duration
  utils.seed_generator(SEED=TRAINING_PARAMETERS['TRAIN_SEED'])
  start_time = timer()
  if(model_type==TRAINING_PARAMETERS['MODEL_TYPES'][1]):
    # Set loss function for UnetS
    #loss_fn = torch.nn.MSELoss() for testing purposes / best is BCEwithLogitsLoss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    logging.info(f"[TRAIN] Loss: {loss_fn}, Optimizer: {type(optimizer).__name__}, Learning Rate: {optimizer.param_groups[0]['lr']}")  
    model_results=engine_encdec.train(model=model,train_dataloader=train_dataloader, test_dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=HYPER_PARAMETERS['NUM_EPOCHS'], device=device, transform=transform)
  else:
    logging.info(f"[TRAIN] Loss: {loss_fn}, Optimizer: {type(optimizer).__name__}, Learning Rate: {optimizer.param_groups[0]['lr']}")  
    model_results=engine.train(model=model,train_dataloader=train_dataloader, test_dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=HYPER_PARAMETERS['NUM_EPOCHS'], device=device, transform=transform)
  end_time = timer()
  print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
  utils.plot_loss_curves(model_results,model_type=model_type, transform=transform)

  



if __name__ == "__main__":
    main()

