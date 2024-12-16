"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import random

def save_model(model: torch.nn.Module, target_dir: str, model_type: str, device="cuda"):
  """Saves a PyTorch model to a target directory.
  Args:
  model: A target PyTorch model to save.
  target_dir: A directory for saving the model to.
  model_name: A filename for the saved model. FileEnding pth will be added
  Example usage:
  save_model(model=model_0, target_dir="model", model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path("model/"+target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)
  # Create model save path
  save_format=".pth"
  model_name = model_type + "_" + device + save_format
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)

def load_model(model: torch.nn.Module, target_dir: str, model_type: str, device="cuda"):
  """Loads a PyTorch model and returns it.
  Args:
  model: A target PyTorch model to load.
  target_dir(str): A directory for loading the model from.
  model_type(str): Filename of the model, which will be loaded. FileEnding ".pth" will be added.
  device(str): Device(cuda/cpu) to be used.
  """
  # Create target directory
  target_dir_path = Path("model/"+target_dir)
  # Create model save path
  save_format=".pth"
  model_name = model_type + "_" + device + save_format
  model_load_path = target_dir_path / model_name
  model.load_state_dict(torch.load(f=model_load_path))
  model = model.to(device)

  # Save the model state_dict()
  print(f"[INFO] Loading model from: {model_load_path}")
  return model


def x():
   load_model()


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


def seed_generator(SEED=16923):                #Random Seed Generator for each function
  seed=random.seed(SEED)
  torch_seed=torch.manual_seed(SEED)
  return seed,torch_seed




def load_data(name):
    """
    Load Dataset out of name of file(train,valid,test)
    
    Parameters:
    name(string) : String with train,valid,test to load their dataset
    
    Returns:
    Dataset of GDM(Gas Distribution Map) and GSL(Gas Source Location) seperately
    """
    dataset = torch.load("data/original/"+name+".pt")
    dataset_GDM=dataset["GDM"]    
    dataset_GSL=dataset["GSL"] 
    return dataset_GDM,dataset_GSL


def save_dataset(dataset_GDM,dataset_GSL,dataset_type,dataset):
  target_dir_path = Path("data/MyTensor")
  target_dir_path.mkdir(parents=True, exist_ok=True)  
  target_dir_path = Path(f"data/MyTensor/datasets_{dataset_type}")
  target_dir_path.mkdir(parents=True, exist_ok=True) 
  torch.save({'X': dataset_GDM, 'y':dataset_GSL},f"data/MyTensor/datasets_{dataset_type}/{dataset}.pt")
  print(f"[INFO] Dataset saved at: data/MyTensor/datasets_{dataset_type}/{dataset}.pt")