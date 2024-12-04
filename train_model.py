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
TRAIN_SEED=16923

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

def main():
  utils.seed_generator(SEED=LOAD_SEED)
  dataloader_types=["Flattened","Distinctive"] #flattened x:30x25 -> y:750, distinctive x:30x25 -> y:30 
  model_types=["VGG24","CNN","VGGVariation"]   #model_types of model_builder -> Simple CNN, VGGVariation(2 Conv Blocks), VGG24(more complex 3 Conv Blocks)
  train_all_models(dataloader_type=dataloader_types[0],model_type= model_types[0])




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
  #train_dataset = create_dataset.Combined_Distinctive_Source(f"data/MyTensor/datasets_{dataloader_type}/train.pt")
  #print(train_dataset)
  #train_dataset=train_transforms(train_dataset.X.numpy())
  #print(train_dataset)
  
  #Set Random Seed
  utils.seed_generator(SEED=LOAD_SEED)
  # Create dataloader and model

  train_dataloader,test_dataloader,valid_dataloader,classes = create_dataloader.create_dataloader(dataloader_type=dataloader_type, batch_size=BATCH_SIZE)
  model = model_builder.choose_model(model_type=model_type,output_shape=classes,device=device)
  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  #Training + Duration
  utils.seed_generator(SEED=TRAIN_SEED)
  start_time = timer()
  mode_results=engine.train(model=model,train_dataloader=train_dataloader, test_dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=NUM_EPOCHS, device=device)
  end_time = timer()
  print(f"Total training time: {end_time-start_time:.3f} seconds")
  # Save the model and plot loss curve
  utils.save_model(model=model,target_dir=dataloader_type,model_type=model_type)
  utils.plot_loss_curves(mode_results)





if __name__ == "__main__":
    main()

