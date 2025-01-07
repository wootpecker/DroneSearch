"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import os

def seed_generator(SEED=16923):                #Random Seed Generator for each function
  seed=random.seed(SEED)
  torch_seed=torch.manual_seed(SEED)
  return seed,torch_seed




def save_dataset(dataset_GDM,dataset_GSL,dataset,augmented=False):
  target_dir_path = Path("data")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  if(augmented):
    target_dir_path = Path(f"data/datasets_tensor_augmented/")
  else:
    target_dir_path = Path(f"data/datasets_tensor/")
  target_dir_path.mkdir(parents=True, exist_ok=True) 
  torch.save({'X': dataset_GDM, 'y':dataset_GSL},f"{target_dir_path}/{dataset}.pt")
  print(f"[INFO] Dataset_GDM shape: {dataset_GDM.shape}")  
  print(f"[INFO] Dataset_GSL shape: {dataset_GSL.shape}")  
  print(f"[SAVE] Dataset saved at: {target_dir_path}\{dataset}.pt")


def load_dataset(dataset_name, augmented=False):
  if(augmented):
    target_dir_path = Path(f"data/datasets_tensor_augmented/")
  else:
    target_dir_path = Path(f"data/datasets_tensor/")
  dataset = torch.load(f"{target_dir_path}\{dataset_name}.pt")
  dataset_GDM = dataset['X']
  dataset_GSL = dataset['y']
  print(f"[INFO] Dataset_GDM shape: {dataset_GDM.shape}")  
  print(f"[INFO] Dataset_GSL shape: {dataset_GSL.shape}")  
  print(f"[LOAD] Dataset was loaded from: {target_dir_path}\{dataset_name}.pt")
  return dataset_GDM,dataset_GSL



def plot_image(image, title=""):
    """
    Plots a single image.
    
    Args:
        image (Tensor): The image to plot.
        title (str): The title of the plot.
    """
    
    plt.imshow(image, cmap='viridis')
    plt.title(title)
    #plt.show()


def save_image(image,index, title=""):
  """
  Saves a single image.
  
  Args:
      image (Tensor): The image to save.
      title (str): The title of the plot.
  """
  title = f"{title}_{index:04d}"
  fig = plot_image(image, title)
  target_dir_path = Path("data")
  target_dir_path.mkdir(parents=True, exist_ok=True)  
  target_dir_path = Path(f"data/images")
  target_dir_path.mkdir(parents=True, exist_ok=True) 
  plt.savefig(f"{target_dir_path}/{title}.png")
  print(f"[SAVE] Image saved at: {target_dir_path}\{title}.png")
  plt.close(fig)
  #fig, ax = plt.subplots()
  #ax.imshow(image, cmap='viridis')
  #ax.set_title(title)
  #plt.close(fig)  # Close the figure to prevent it from displaying
  #return fig


def plot_more_images(images, title=""):
    """
    Plots multiple images.
    
    Args:
        images (Tensor): The images to plot.
        title (str): The title of the plot.
    """
    if(len(images)<6):
      fig, axes = plt.subplots(len(images), 1, figsize=(15, 18))
    else:      
      fig, axes = plt.subplots(6, math.ceil(len(images)/6), figsize=(15, 18))
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
          break
        ax.imshow(images[i], cmap='viridis')        
        #ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_model(model: torch.nn.Module, model_type: str, epoch=None, device="cuda"):
  """Saves a PyTorch model to a target directory.
  Args:
  model: A target PyTorch model to save.
  target_dir: A directory for saving the model to.
  model_name: A filename for the saved model. FileEnding pth will be added
  Example usage:
  save_model(model=model_0, target_dir="model", model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(f"model")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"model/{model_type}")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  # Create model save path
  save_format=".pth"
  if epoch is None:
    model_name = model_type + "_" + device + save_format
  else:
    model_name = model_type + "_" + device + f"_{epoch:03d}" + save_format
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[SAVE] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)



def load_model(model: torch.nn.Module, model_type: str, device="cuda"):
  """Saves a PyTorch model to a target directory.
  Args:
  model: A target PyTorch model to save.
  target_dir: A directory for saving the model to.
  model_name: A filename for the saved model. FileEnding pth will be added
  Example usage:
  save_model(model=model_0, target_dir="model", model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(f"model")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"model/{model_type}")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  files=os.listdir(target_dir_path)
  
  if len(files)==0:
    return model,0
  start=len(files)
  save_format=".pth"
  model_name = model_type + "_" + device + f"_{start:03d}" + save_format
  model_load_path = target_dir_path / model_name
  model.load_state_dict(torch.load(f=model_load_path))
  model = model.to(device)

  # Save the model state_dict()
  print(f"[LOAD] Loading model from: {model_load_path}")
  return model,start




    
def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()




