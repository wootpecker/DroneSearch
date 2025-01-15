"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import model_dataloader, utils, engine, model_builder,engine_encdec
from timeit import default_timer as timer 
from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001
LOAD_SEED=16923
TRAIN_SEED=42

MODEL_TYPES = ["VGG", "EncoderDecoder", "VGGVariation"]
DATASET_TYPES = ["01_Winter", "02_Spring", "03_Summer", "04_Autumn"] 


# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
  model_type=MODEL_TYPES[1]
  dataloader_type=DATASET_TYPES[0]
  transform=True

  utils.reset_training(model_type=model_type)  
  train_all_models(dataloader_type=dataloader_type, model_type=model_type, transform=transform)
   







    

def train_all_models(dataloader_type="01_Winter", model_type= "VGG", transform=True):
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
  utils.seed_generator(SEED=LOAD_SEED)

  # Create dataloader and model
  train_dataloader,test_dataloader,classes = model_dataloader.create_dataloader(model_type=model_type, batch_size=BATCH_SIZE, transform=transform)
  model = model_builder.choose_model(model_type=model_type,output_shape=classes,device=device)

  # Set loss and optimizer
  # either BCEwithLogits
  #loss_fn = torch.nn.BCELoss()          # or     BCE with sigmoid in Model 
  #print(f"{model.}")

  loss_fn = torch.nn.CrossEntropyLoss()     
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  
  #Training + Duration
  utils.seed_generator(SEED=TRAIN_SEED)
  start_time = timer()
  if(model_type==MODEL_TYPES[1]):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    print(f"[TRAIN] loss_fn: {loss_fn}, optimizer: {optimizer.__str__}")
    mode_results=engine_encdec.train(model=model,train_dataloader=train_dataloader, test_dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=NUM_EPOCHS, device=device)
  else:
    print(f"[TRAIN] loss_fn: {loss_fn}, optimizer: {optimizer.__str__}")
    mode_results=engine.train(model=model,train_dataloader=train_dataloader, test_dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=NUM_EPOCHS, device=device)
  end_time = timer()
  print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
  # Save the model and plot loss curve
  #utils.save_model(model=model,model_type=model_type,device=device)
  utils.plot_loss_curves(mode_results)

  



if __name__ == "__main__":
    main()

