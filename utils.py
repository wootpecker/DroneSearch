"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import random
import math
import os
import numpy
import shutil
import logging

MODEL_TYPES = ["VGG", "UnetS", "VGGVariation"]




def main():
  #reset_all()
  pass

def seed_generator(SEED=16923):                #Random Seed Generator for each function
  seed=random.seed(SEED)
  torch_seed=torch.manual_seed(SEED)
  numpy_seed = numpy.random.seed(SEED)
  return seed,torch_seed,numpy_seed




def save_dataset(dataset_GDM,dataset_GSL,dataset,augmented=False):
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
    plt.title(title)
    plt.show()


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
  logging.info(f"[SAVE] Image saved at: {target_dir_path}\{title}.png")  
  plt.close(fig)
  #fig, ax = plt.subplots()
  #ax.imshow(image, cmap='viridis')
  #ax.set_title(title)
  #plt.close(fig)  # Close the figure to prevent it from displaying
  #return fig


def plot_more_images(images, title="", save=False):
    """
    Plots multiple images.
    
    Args:
        images (Tensor): The images to plot.
        title (str): The title of the plot.
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
        #ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    
    if save:
      target_dir_path = Path(f"data/images")
      target_dir_path.mkdir(parents=True, exist_ok=True) 
      plt.savefig(f"{target_dir_path}/{title}.png")
      logging.info(f"[SAVE] Image saved at: {target_dir_path}\{title}.png")        
    else:
      plt.show()


def save_model(model: torch.nn.Module, model_type: str, epoch=None, device="cuda", transform=True):
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
  """Saves a PyTorch model to a target directory.
  Args:
  model: A target PyTorch model to save.
  target_dir: A directory for saving the model to.
  model_name: A filename for the saved model. FileEnding pth will be added
  Example usage:
  save_model(model=model_0, target_dir="model", model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(f"data/random_state")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"data/random_state/{model_type}")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  rng_state_dict = {
  'cpu_rng_state': torch.get_rng_state(),
  'gpu_rng_state': torch.cuda.get_rng_state(),
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
  """Saves a PyTorch model to a target directory.
  Args:
  model: A target PyTorch model to save.
  target_dir: A directory for saving the model to.
  model_name: A filename for the saved model. FileEnding pth will be added
  Example usage:
  save_model(model=model_0, target_dir="model", model_name="05_going_modular_tingvgg_model.pth")
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
  torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
  numpy.random.set_state(rng_state_dict['numpy_rng_state'])
  random.setstate(rng_state_dict['py_rng_state'])
  # Save the model state_dict()
  logging.info(f"[LOAD] Loading Random State from: {model_load_path}")  
  return start

    
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



def save_loss(results,model_type: str, device="cuda"):
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
  """Saves a PyTorch model to a target directory.
  Args:
  model: A target PyTorch model to save.
  target_dir: A directory for saving the model to.
  model_name: A filename for the saved model. FileEnding pth will be added
  Example usage:
  save_model(model=model_0, target_dir="model", model_name="05_going_modular_tingvgg_model.pth")
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

def reset_training(model_type:str,transform=True):
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



if __name__ == "__main__":
    main()