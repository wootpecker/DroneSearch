"""
utils.py

This module provides utility functions to support PyTorch-based model training, evaluation, and reproducibility.
It includes helpers for dataset saving/loading, model checkpointing, random state saving/loading, plotting, and training reset.
Part of this code has been implemented from: https://www.learnpytorch.io/05_pytorch_going_modular/

-----------------------------
Constants:
- MODEL_TYPES: List of supported model architectures.
- device: Target device for computations (CPU or CUDA).

-----------------------------
Functions:
- main():
    Used for testing.

- seed_generator(SEED)
  Sets seeds for Python, NumPy, and PyTorch random number generators for reproducibility.

- save_dataset(dataset_GDM, dataset_GSL, dataset, augmented):
  Saves PyTorch tensor datasets (gas distribution map and gas source location) to disk, supporting both original and augmented datasets.

- load_dataset(dataset_name, augmented):
  Loads PyTorch tensor datasets from disk, supporting both original and augmented datasets.

- plot_image(image, title):
  Plots a single image tensor using matplotlib.

- save_image(image, index, title):
  Saves a single image tensor as a PDF file.

- plot_more_images(images, title, save):
  Plots multiple image tensors in a grid, with optional saving as a PDF.

- save_model(model, model_type, epoch, device, transform):
  Saves a PyTorch model's state_dict to an organized directory structure, supporting original and transformed data.

- load_model(model, model_type, device, transform):
  Loads a PyTorch model's state_dict from a checkpoint, returning the model and the number of epochs trained.

- save_random(model_type, epoch, device):
  Saves the random state (PyTorch, NumPy, Python) to disk for reproducibility.

- load_random(model_type, epoch, device):
  Loads and restores the random state from disk for reproducibility.

- plot_loss_curves(results, model_type, transform):
  Plots and saves training/testing loss and accuracy curves from a results dictionary.

- save_loss(results, model_type, device):
  Saves a dictionary of training/testing loss and accuracy metrics to disk.

- load_loss(model_type, device):
  Loads a dictionary of training/testing loss and accuracy metrics from disk.

- reset_training(model_type, transform):
  Removes results folders for a specific model and data type (original or transformed) to allow clean restarts.

- reset_all():
  Removes results folders for all supported model types and both original and transformed data.

- compare_rng_states(rng_state_dictin, rng_state_dict):
  Compares two random state dictionaries for debugging reproducibility issues.

- test_random_state_differences():
  Utility function to compare consecutive saved random states for a model.

-----------------------------
Dependencies:
- torch, pathlib, typing, matplotlib, numpy, random, math, os, shutil, logging

-----------------------------
Usage:
- Import this module in your training or evaluation scripts to access utility functions for data handling, checkpointing, reproducibility, and visualization.
"""



import torch
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy
import random
import math
import os
import shutil
import logging

MODEL_TYPES = ["VGG8", "UnetS"]
device = "cuda" if torch.cuda.is_available() else "cpu"




def main():
  #reset_all()
  pass

def seed_generator(SEED=16923):                #Random Seed Generator for each function
  """
  Random Seed Generator for reproducibility.
  
  Args:
      SEED (int): The seed value to set for random number generation. Default is 16923.
  
  Returns:
      Tuple: A tuple containing the random seed, torch seed, and numpy seed.
  """
  seed=random.seed(SEED)
  torch_seed=torch.manual_seed(SEED)
  numpy_seed = numpy.random.seed(SEED)
  return seed,torch_seed,numpy_seed




def save_dataset(dataset_GDM,dataset_GSL,dataset,augmented=False):
  """
  Saves tensor dataset.
  
  Args:
      dataset_GDM (Tensor): The gas distribution map dataset to save.
      dataset_GSL (Tensor): The gas source location dataset to save.
      dataset (str): The name of the seasonal dataset to save.
      augmented (bool): If the augmented dataset should be loaded. Implementation utilizes Data Augmentation before training, due to long loading times of big augmented datasets.
  """
  target_dir_path = Path("data")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  if(augmented):
    target_dir_path = Path(f"data/datasets_tensor_augmented/")
  else:
    target_dir_path = Path(f"data/datasets_tensor/")
  target_dir_path.mkdir(parents=True, exist_ok=True) 
  torch.save({'X': dataset_GDM, 'y':dataset_GSL},f"{target_dir_path}/{dataset}.pt")
  logging.info(f"Dataset_GDM shape: {dataset_GDM.shape}")
  logging.info(f"Dataset_GSL shape: {dataset_GSL.shape}")
  logging.info(f"[SAVE] Dataset saved at: {target_dir_path}\{dataset}.pt")



def load_dataset(dataset_name, augmented=False):
  """
  Loads tensor dataset.
  
  Args:
      dataset_name (str): The name of the seasonal dataset to load.
      augmented (bool): If the augmented dataset should be loaded. Implementation utilizes Data Augmentation before training, due to long loading times of big augmented datasets.
  
  Returns:
      Tuple: A tuple containing the gas distribution map dataset and the gas source location dataset.
  """
  if(augmented):
    target_dir_path = Path(f"data/datasets_tensor_augmented/")
  else:
    target_dir_path = Path(f"data/datasets_tensor/")
  dataset = torch.load(f"{target_dir_path}\{dataset_name}.pt",weights_only=True)
  dataset_GDM = dataset['X']
  dataset_GSL = dataset['y']
  logging.info(f"Dataset_GDM shape: {dataset_GDM.shape}")
  logging.info(f"Dataset_GSL shape: {dataset_GSL.shape}")
  logging.info(f"[LOAD] Dataset was loaded from: {target_dir_path}\{dataset_name}.pt")  
  return dataset_GDM,dataset_GSL



def plot_image(image, title=""):
    """
    Plots a single image.
    
    Args:
        image (Tensor): The image to plot.
        title (str): The title of the plot.
    """
    image=image.squeeze().unsqueeze(-1)
    plt.imshow(image, cmap='viridis', origin='lower')
    plt.show()


def save_image(image,index, title=""):
  """
  Saves a single image.
  
  Args:
      image (Tensor): The image to save.
      index (int): The index of the image for naming.
      title (str): The title of the plot.
  """
  title = f"{title}_{index:04d}"
  fig = plot_image(image, title)
  target_dir_path = Path("data")
  target_dir_path.mkdir(parents=True, exist_ok=True)  
  target_dir_path = Path(f"data/images")
  target_dir_path.mkdir(parents=True, exist_ok=True) 
  plt.savefig(f"{target_dir_path}/{title}.pdf")
  logging.info(f"[SAVE] Image saved at: {target_dir_path}\{title}.pdf")  
  plt.close(fig)



def plot_more_images(images, title="", save=False):
    """
    Plots multiple images.
    
    Args:
        images (Tensor): The images to plot.
        title (str): The title of the plot.
        save (bool): Whether to save the plot as a PDF file.
    """
    images=images.squeeze().unsqueeze(-1)

    if(len(images)<6):
      fig, axes = plt.subplots(len(images), 1, figsize=(15, 18))
    else:      
      fig, axes = plt.subplots(6, math.ceil(len(images)/6), figsize=(15, 18))
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
          break
        ax.imshow(images[i], cmap='viridis', origin='lower')        
    plt.suptitle(title)
    plt.tight_layout()
    
    if save:
      target_dir_path = Path(f"data/images")
      target_dir_path.mkdir(parents=True, exist_ok=True) 
      plt.savefig(f"{target_dir_path}/{title}.pdf")
      logging.info(f"[SAVE] Image saved at: {target_dir_path}\{title}.pdf")        
    else:
      plt.show()


def save_model(model: torch.nn.Module, model_type: str, epoch=None, device="cuda", transform=True):
  """
  Saves a PyTorch model to a target directory.
  Args:
    model: A target PyTorch model to save.
    model_type: A string indicating the type of model (e.g., "VGG8", "UnetS").
    epoch: An integer indicating the current epoch (optional, to load during subsequent training).
    device: A target device to compute on (e.g. "cuda" or "cpu").
    transform: A boolean indicating if the model is trained on transformed data.
  """
  # Create target directory
  target_dir_path = Path(f"model")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"model/original")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"model/original/{model_type}")
  target_dir_path.mkdir(parents=True, exist_ok=True)  
  if transform:
    target_dir_path = Path(f"model/transform")
    target_dir_path.mkdir(parents=True, exist_ok=True)    
    target_dir_path = Path(f"model/transform/{model_type}")
    target_dir_path.mkdir(parents=True, exist_ok=True)
  # Create model save path
  save_format=".pth"
  if epoch is None:
    model_name = model_type + "_" + device + save_format
  else:
    model_name = model_type + "_" + device + f"_{epoch:03d}" + save_format
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  logging.info(f"[SAVE] Saving model to: {model_save_path}")       
  torch.save(obj=model.state_dict(), f=model_save_path)



def load_model(model: torch.nn.Module, model_type: str, device="cuda", transform=True):
  """
  Loads a PyTorch model from target directory.
  Args:
    model: A target PyTorch model to load.
    model_type: A string indicating the type of model (e.g., "VGG8", "UnetS").
    device: A target device to compute on (e.g. "cuda" or "cpu").
    transform: A boolean indicating if the model is trained on transformed data.
  Returns:
    model: The loaded PyTorch model with its state_dict.
    start: An integer indicating the number of epochs the model has been trained for.
  
  """
  # Create target directory
  target_dir_path = Path(f"model")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"model/original")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"model/original/{model_type}")
  target_dir_path.mkdir(parents=True, exist_ok=True)  
  model_setting="original"
  if transform:
    target_dir_path = Path(f"model/transform")
    target_dir_path.mkdir(parents=True, exist_ok=True)    
    target_dir_path = Path(f"model/transform/{model_type}")
    target_dir_path.mkdir(parents=True, exist_ok=True)
    model_setting="transformed"
  files=os.listdir(target_dir_path)
  
  if len(files)==0:
    return model,0
  elif len(files)==1:
    model_load_path = target_dir_path / files[0]
    start=1
  else:
    start=len(files)
    save_format=".pth"
    model_name = model_type + "_" + device + f"_{start:03d}" + save_format
    model_load_path = target_dir_path / model_name
  model.load_state_dict(torch.load(f=model_load_path,weights_only=True))
  model = model.to(device)

  # Save the model state_dict()
  logging.info(f"[LOAD] Loading {model_setting} model from: {model_load_path}")
  return model,start




def save_random(model_type: str, epoch=None, device="cuda"):
  """
  Saves a the random state to a target directory.
  Args:
    model_type: A string indicating the type of model (e.g., "VGG8", "UnetS").
    epoch: An integer indicating the current epoch (optional, to load during subsequent training).
    device: A target device to compute on (e.g. "cuda" or "cpu").
  """
  # Create target directory
  target_dir_path = Path(f"data/random_state")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"data/random_state/{model_type}")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  # Create random state dictionary
  if device == "cuda":
    rng_state_dict = {
      'cpu_rng_state': torch.get_rng_state(),
      'gpu_rng_state': torch.cuda.get_rng_state(),
      'numpy_rng_state': numpy.random.get_state(),  
      'py_rng_state': random.getstate()
    }
  else:
    rng_state_dict = {
      'cpu_rng_state': torch.get_rng_state(),
      'numpy_rng_state': numpy.random.get_state(),  
      'py_rng_state': random.getstate()
    }

  # Create model save path
  save_format=".ckpt"

  model_name = model_type + "_" + device + f"_{epoch:03d}" + save_format
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  logging.info(f"[SAVE] Saving Random State to: {model_save_path}")
  torch.save(rng_state_dict, f=model_save_path)



def load_random(model_type: str, epoch=None, device="cuda"):
  """
  Loads the last random state from target directory.
  Args:
    model_type: A string indicating the type of model (e.g., "VGG8", "UnetS").
    epoch: An integer indicating the current epoch (optional, to load during subsequent training).
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Returns:
    start: An integer indicating the number of epochs the model has been trained for.
  """
  # Create target directory
  target_dir_path = Path(f"data/random_state")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"data/random_state/{model_type}")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  files=os.listdir(target_dir_path)
  
  if len(files)==0:
    return 0
  start=len(files)
  save_format=".ckpt"
  model_name = model_type + "_" + device + f"_{start:03d}" + save_format
  if epoch:
    model_name = model_type + "_" + device + f"_{epoch:03d}" + save_format
  model_load_path = target_dir_path / model_name


  rng_state_dict=torch.load(f=model_load_path)
  torch.set_rng_state(rng_state_dict['cpu_rng_state'])
  if device == "cuda":
    torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
  numpy.random.set_state(rng_state_dict['numpy_rng_state'])
  random.setstate(rng_state_dict['py_rng_state'])
  # Save the model state_dict()
  logging.info(f"[LOAD] Loading Random State from: {model_load_path}")  
  return start

    
def plot_loss_curves(results: Dict[str, List[float]], model_type=MODEL_TYPES[0], transform=True):
  """
  Plots training curves of a results dictionary.
  Args:
    results (dict): dictionary containing list of values, e.g.
        {"train_loss": [...],
         "train_acc": [...],
         "test_loss": [...],
         "test_acc": [...]}
    model_type (str): type of model used for training, e.g. "VGG8" or "UnetS".
    transform (bool): whether the model was trained on transformed data or not.
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
  plt.figure(figsize=(11, 5))
  # Plot loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label='Training Set')
  plt.plot(epochs, test_loss, label='Testing Set')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend()
  # Plot accuracy
  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, label='Training Set')
  plt.plot(epochs, test_accuracy, label='Testing Set')
  plt.ylabel('Accuracy')
  plt.xlabel('Epochs')
  plt.legend()
  target_dir_path = Path(f"results/loss_curve")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  save_format=".pdf"
  transform_str="on_original_data"
  if transform:
    transform_str="on_transformed_data"
  file_name = "loss_" + model_type + "_" + transform_str + save_format
  file_save_path = target_dir_path / file_name
  plt.savefig(file_save_path)
  plt.show()



def save_loss(results,model_type: str, device="cuda"):
  """
  Saves the result dictionary to a target directory.
  Args:
    results: A dictionary containing training and testing loss and accuracy metrics.
             In the form: {train_loss: [...],
                           train_acc: [...],
                           test_loss: [...],
                           test_acc: [...]}
    model_type: A string indicating the type of model (e.g., "VGG8", "UnetS").
    device: A target device to compute on (e.g. "cuda" or "cpu").
  """
  # Create target directory
  target_dir_path = Path(f"data/loss_curve")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"data/loss_curve/{model_type}")
  target_dir_path.mkdir(parents=True, exist_ok=True)


  # Create save path
  save_format=".ckpt"
  file_name = model_type + "_" + device + save_format
  file_save_path = target_dir_path / file_name

  # Save the Loss
  logging.info(f"[SAVE] Saving Loss to: {file_save_path}")  
  torch.save(obj=results, f=file_save_path)



def load_loss(model_type: str, device="cuda"):
  """
  Saves the result dictionary to a target directory.
  Args:
    model_type: A string indicating the type of model (e.g., "VGG8", "UnetS").
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Returns:
   results: A dictionary containing training and testing loss and accuracy metrics.
             In the form: {train_loss: [...],
                           train_acc: [...],
                           test_loss: [...],
                           test_acc: [...]}
  """
  # Create empty results dictionary
  results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
  }
  # Create target directory
  target_dir_path = Path(f"data/loss_curve")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"data/loss_curve/{model_type}")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  files=os.listdir(target_dir_path)
  if len(files)==0:
    return results
  save_format=".ckpt"
  file_name = model_type + "_" + device + save_format
  file_load_path = target_dir_path / file_name


  results=torch.load(f=file_load_path,weights_only=True)
  logging.info(f"[LOAD] Loading Loss Results from: {file_load_path}")  

  return results



def reset_training(model_type:str,transform=True):
  """
  Saves the result dictionary to a target directory.
  Args:
    model_type: A string indicating the type of model (e.g., "VGG8", "UnetS").
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Returns:
   results: A dictionary containing training and testing loss and accuracy metrics.
             In the form: {train_loss: [...],
                           train_acc: [...],
                           test_loss: [...],
                           test_acc: [...]}
  """
  folder_dir_path=[]
  folder_dir_path.append(Path(f"data/random_state/{model_type}"))
  folder_dir_path.append(Path(f"data/loss_curve/{model_type}"))
  if transform:
    folder_dir_path.append(Path(f"model/transform/{model_type}"))
  else:
    folder_dir_path.append(Path(f"model/original/{model_type}"))
  for folder in folder_dir_path:
    try:
        shutil.rmtree(folder)
        logging.info(f"[RESET] Folder '{folder}' deleted successfully.")  
    except FileNotFoundError:
        logging.info(f"[RESET] Folder '{folder}' does not exist.")  
    except PermissionError:
        logging.info(f"[RESET] Permission denied to delete '{folder}'.") 
    except Exception as e:   
        logging.info(f"[RESET] An error occurred: {e}.") 

def reset_all():
  for model in MODEL_TYPES:
    reset_training(model, transform=True)
    reset_training(model, transform=False)












####             HELPER

def compare_rng_states(rng_state_dictin, rng_state_dict):
  print("\n--------------------------------")
  if torch.equal(rng_state_dictin['cpu_rng_state'], rng_state_dict['cpu_rng_state']):
    print(f"['cpu_rng_state'] TRUE,")
  else:
    print(f"['cpu_rng_state'] FALSE, ")
  
  if torch.equal(rng_state_dictin['gpu_rng_state'], rng_state_dict['gpu_rng_state']):
    print(f"['gpu_rng_state'] TRUE,")
  else:
    print(f"['gpu_rng_state'] FALSE, ")
  state1 = rng_state_dictin['numpy_rng_state']
  state2 = rng_state_dict['numpy_rng_state']
  #are_states_equal = state1 == state2
  #are_states_equal = rng_state_dictin['numpy_rng_state'] == rng_state_dict['numpy_rng_state']
  are_states_equal = (
    state1[0] == state2[0]  # The generator type (e.g., "MT19937")
    and numpy.array_equal(state1[1], state2[1])  # The state array
    and state1[2:] == state2[2:]  # Other elements (e.g., position)
)
  if (are_states_equal):
    print(f"['numpy_rng_state'] TRUE,")
  else:
    print(f"['numpy_rng_state'] FALSE, ")
  
  if rng_state_dictin['py_rng_state'] == rng_state_dict['py_rng_state']:
    print(f"['py_rng_state'] TRUE, \n")
  else:
    print(f"['py_rng_state'] FALSE, \n")

      


def test_random_state_differences():
  model_type="VGG"
  device="cuda"
  target_dir_path = Path(f"data/random_state")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"data/random_state/{model_type}")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  files=os.listdir(target_dir_path)

  save_format=".ckpt"
  for start in range(2, len(files)+1):
    print(f"elements: {start-1}:{start}")
    model_name = model_type + "_" + device + f"_{start:03d}" + save_format
    model_namefirst = model_type + "_" + device + f"_{start-1:03d}" + save_format    
    model_load_path = target_dir_path / model_name
    model_loadfirst_path = target_dir_path / model_namefirst
    state1=torch.load(f=model_load_path,weights_only=True)
    state2=torch.load(f=model_loadfirst_path,weights_only=True)
    compare_rng_states(state1,state2)
    print("--------------------------------")





if __name__ == "__main__":
    main()