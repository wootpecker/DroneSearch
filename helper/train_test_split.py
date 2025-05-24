"""
train_test_split.py

This module provides helper functions to split datasets into training and testing sets for training of machine learning models.
It includes utilities for reshaping input data, random shuffling, and converting coordinate labels into grid representations suitable for model training.

-----------------------------
Functions:
- main(): Used for testing.

- load_and_split_dataset(datasets_GDM, datasets_GSL, train_ratio=0.8, save=True, window_size=[64,64]):
    Splits the provided datasets into training and testing sets based on the specified ratio.
    Optionally saves the resulting splits using utility functions.

- coordinates_to_grid(dataset_GSL, window_size=[64,64]):
    Converts coordinate labels into grid (one-hot) tensors for use in grid-based models.

-----------------------------
Dependencies:
- numpy, torch, matplotlib, logging
- Custom modules: utils

-----------------------------
Usage:
- Import and call `load_and_split_dataset` with your data arrays to obtain train/test splits.
"""



import torch
from torch.utils import data
from timeit import default_timer as timer 
import logging

import sys
sys.path.append("")
sys.path.append("..")
import utils



def main():
    pass


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

    X = X.reshape(-1,1,window_size[0],window_size[1])
    y = y.reshape(-1,2)

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
    
    train_GSL = coordinates_to_grid(train_GSL,window_size=window_size)
    test_GSL = coordinates_to_grid(test_GSL,window_size=window_size)

    logging.info(f"X Train shape: {train_GDM.shape}, y Train shape: {train_GSL.shape}")
    logging.info(f"X Test shape: {test_GDM.shape}, y Test shape: {test_GSL.shape}")  
    if save:
        utils.save_dataset(train_GDM, train_GSL, "train", augmented=True)
        utils.save_dataset(test_GDM, test_GSL, "test", augmented=True)
    return train_GDM,train_GSL,test_GDM,test_GSL
    

def coordinates_to_grid(dataset_GSL, window_size=[64,64]):
    grid = torch.zeros((dataset_GSL.shape[0],1, window_size[0], window_size[1]))
    for i in range(dataset_GSL.shape[0]):
        x, y = dataset_GSL[i]
        grid[i,0, x, y] = 1
    return torch.FloatTensor(grid)

if __name__ == "__main__":
    main()
