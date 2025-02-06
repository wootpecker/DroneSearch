"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import logging
from logs import logger
import model_dataloader, utils, engine, model_builder,engine_encdec
from timeit import default_timer as timer 
from torchvision import transforms

# Setup hyperparameters


MODEL_TYPES = ["VGG", "EncoderDecoder", "VGGVariation"]


HYPER_PARAMETERS = {
              "SAVE_DATASET": False,
               "TRANSFORM": True,
               "MODEL_TYPES": ["VGG", "EncoderDecoder", "VGGVariation"],
               "LOGS_SAVE": True,
               "AMOUNT_SAMPLES": 16,
               "WINDOW_SIZE": [64,64]
  }



TRAINING_PARAMETERS = {
              "NUM_EPOCHS": 50,
               "BATCH_SIZE": 32,
               "LEARNING_RATE": 0.001,
               "LOAD_SEED": 16923,
               "TRAIN_SEED": 42
  }




# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
  model_type=HYPER_PARAMETERS['MODEL_TYPES'][2]
  transform=HYPER_PARAMETERS['TRANSFORM']
  logs_save=HYPER_PARAMETERS['LOGS_SAVE']
  logger.logging_config(logs_save=logs_save,amount_samples=HYPER_PARAMETERS['AMOUNT_SAMPLES'], transform=transform, model_type=model_type, window_size=HYPER_PARAMETERS['WINDOW_SIZE'])

  utils.reset_training(model_type=model_type)  
  train_all_models(model_type=model_type, transform=transform)
   







    

def train_all_models(model_type= "VGG", transform=True):
  """Trains a PyTorch model, saving it and create a loss curve plot.
s
    Args:
    model(string): Type of model to be used for training.
    dataloader(string): Type of dataset to be used for training.

    Returns:
    Saved model in target dir.
    Plot of loss curve
    """
  
  #Set Random Load Seed
  utils.seed_generator(SEED=TRAINING_PARAMETERS['LOAD_SEED'])

  # Create dataloader and model
  train_dataloader,test_dataloader,classes = model_dataloader.create_dataloader(model_type=model_type, batch_size=TRAINING_PARAMETERS['BATCH_SIZE'], transform=transform, amount_samples=HYPER_PARAMETERS['AMOUNT_SAMPLES'], window_size=HYPER_PARAMETERS['WINDOW_SIZE'])
  model = model_builder.choose_model(model_type=model_type,output_shape=classes,device=device, window_size=HYPER_PARAMETERS['WINDOW_SIZE'])

  # Set loss and optimizer
  # either BCEwithLogits
  #loss_fn = torch.nn.BCELoss()          # or     BCE with sigmoid in Model 
  #print(f"{model.}")

  loss_fn = torch.nn.CrossEntropyLoss()     
  optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMETERS['LEARNING_RATE'])
  
  #Training + Duration
  utils.seed_generator(SEED=TRAINING_PARAMETERS['TRAIN_SEED'])
  start_time = timer()
  if(model_type==MODEL_TYPES[1]):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    logging.info(f"[TRAIN] Loss: {loss_fn}, Optimizer: {type(optimizer).__name__}, Learning Rate: {optimizer.param_groups[0]['lr']}")  
    mode_results=engine_encdec.train(model=model,train_dataloader=train_dataloader, test_dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=TRAINING_PARAMETERS['NUM_EPOCHS'], device=device)
  else:
    logging.info(f"[TRAIN] Loss: {loss_fn}, Optimizer: {type(optimizer).__name__}, Learning Rate: {optimizer.param_groups[0]['lr']}")  
    mode_results=engine.train(model=model,train_dataloader=train_dataloader, test_dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=TRAINING_PARAMETERS['NUM_EPOCHS'], device=device)
  end_time = timer()
  print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
  # Save the model and plot loss curve
  #utils.save_model(model=model,model_type=model_type,device=device)
  utils.plot_loss_curves(mode_results)

  



if __name__ == "__main__":
    main()

