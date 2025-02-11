import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
from torch.utils import data
from timeit import default_timer as timer 
import logging



def main():
    pass
    #load_and_split_dataset()


def load_and_split_dataset(datasets_GDM,datasets_GSL, train_ratio=0.8, save=True, window_size=[64,64]):
    """
    Loads the dataset and splits it into train and test sets.
    
    Args:
        data_path (str): Path to the dataset file.
        train_ratio (float): Ratio of the dataset to use for training.
        augmented (bool): Whether to apply augmen++tation.
        
    Returns:
        train_dataset (Dataset): Training dataset.
        test_dataset (Dataset): Testing dataset.
    """
    X = datasets_GDM
    y = datasets_GSL
    #X = data['X']
    #y = data['y']
    X = X.reshape(-1,1,window_size[0],window_size[1])
    y = y.reshape(-1,2)
    # Calculate the number of training samples
    #print(f"[INFO] X: {X.shape}, y: {y.shape}")

    # Random Seed
    utils.seed_generator()
    
    # Split the dataset
    logging.info(f"Dataset loaded and split into train and test sets.")

    train_size = int(train_ratio * X.shape[0])
    indices = torch.randperm(X.shape[0])
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_GDM = X[train_indices, :, :, :]
    test_GDM = X[test_indices, :, :, :]
    
    train_GSL = y[train_indices, :]
    test_GSL = y[test_indices, :]

    
    
    # Create SuperDataset instances
    train_GSL = coordinates_to_grid(train_GSL,window_size=window_size)
    test_GSL = coordinates_to_grid(test_GSL,window_size=window_size)


    logging.info(f"X Train shape: {train_GDM.shape}, y Train shape: {train_GSL.shape}")
    logging.info(f"X Test shape: {test_GDM.shape}, y Test shape: {test_GSL.shape}")  
    #return train_GDM, train_GSL, test_GDM, test_GSL
    if save:
        utils.save_dataset(train_GDM, train_GSL, "train", augmented=True)
        utils.save_dataset(test_GDM, test_GSL, "test", augmented=True)
    return train_GDM,train_GSL,test_GDM,test_GSL
    

def coordinates_to_grid(dataset_GSL, window_size=[64,64]):
    """
    Converts the coordinates to a grid.
    
    Args:
        y (Tensor): The coordinates.
        
    Returns:
        grid (Tensor): The grid.
    """
    grid = torch.zeros((dataset_GSL.shape[0],1, window_size[0], window_size[1]))
    for i in range(dataset_GSL.shape[0]):
        x, y = dataset_GSL[i]
        grid[i,0, x, y] = 1
    return torch.FloatTensor(grid)

if __name__ == "__main__":
    main()
