"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import create_dataloader, utils, engine, model_builder,create_dataset
from timeit import default_timer as timer 
from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LOAD_SEED=16923
TRAIN_SEED=42
DATASET_TYPES=["Distinctive","Flattened","S-Shape", "Grid", "Random", "Edge","EncoderDecoder"]
MODEL_TYPES=["VGG24","CNN","VGGVariation","UnetEncoderDecoder"] #model_types of model_builder -> Simple CNN, VGGVariation(2 Conv Blocks), VGG24(more complex 3 Conv Blocks)

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

def main():
  utils.seed_generator(SEED=LOAD_SEED)
   
  train_all_models(dataloader_type=DATASET_TYPES[6],model_type= MODEL_TYPES[3])




def train_all_models(dataloader_type="Flattened",model_type= "VGG24"):
  """Trains a PyTorch model, saving it and create a loss curve plot.
s
    Args:
    model(string): Type of model to be used for training.
    dataloader(string): Type of dataset to be used for training.

    Returns:
    Saved model in target dir.
    Plot of loss curve
    """
  train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
  
  #Set Random Load Seed
  utils.seed_generator(SEED=LOAD_SEED)

  # Create dataloader and model
  train_dataloader,test_dataloader,valid_dataloader,classes = create_dataloader.create_dataloader(dataloader_type=dataloader_type, batch_size=BATCH_SIZE)
  model = model_builder.choose_model(model_type=model_type,output_shape=classes,device=device)

  # Set loss and optimizer
  if(dataloader_type==DATASET_TYPES[6]):
    loss_fn = torch.nn.BCEWithLogitsLoss()
  else:
    loss_fn = torch.nn.CrossEntropyLoss()     
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  #Training + Duration
  utils.seed_generator(SEED=TRAIN_SEED)
  start_time = timer()
  mode_results=engine.train(model=model,train_dataloader=train_dataloader, test_dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=NUM_EPOCHS, device=device)
  end_time = timer()
  print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
  # Save the model and plot loss curve
  utils.save_model(model=model,target_dir=dataloader_type,model_type=model_type)
  utils.plot_loss_curves(mode_results)





if __name__ == "__main__":
    main()

