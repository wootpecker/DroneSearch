"""
augment_dataset.py
This helper module provides functionality for augmenting datasets of gas distribution maps and gas source locations for training machine learning models. 
The augmentation process involves extracting sliding windows from original simulation datasets, generating new samples by shifting the gas source location within each window, and stacking the resulting data for use in machine learning models.

-----------------------------
Testing Parameters:
- SAVE_LOGS (bool): Whether to save or show logs during dataset initialization and processing.
- LOGS_FILENAME (str): Filename for saving logs.
- AMOUNT_SAMPLES (int): Number of samples to generate for each gas source location during augmentation.
- WINDOW_SIZE (list of int): Size of the window to use for data augmentation.

Constants:
- SOURCE_DIR: Directory containing dataset of Nicolas Winkler.
- SIMULATIONS: List of available simulation datasets in the source directory.

-----------------------------
Functions:
- main(): Used for testing purposes.

- create_augmented_dataset(amount_samples, window_size): 
    Processes all simulations, augments datasets, and concatenates results.

- augment_datasets(dataset, amount_samples, window_size): 
    Augments a single simulation dataset by extracting windows and shifting source locations.

- extract_samples(dataset, amount_samples): 
    Randomly selects sample indices from a dataset.

- extract_window(tensor, window_size, source_location, new_location): 
    Extracts a window from the tensor, shifting the gas source location.

- create_mapping(all_coordinates, window_size): 
    Generates a mapping of original to new gas source coordinates for window extraction.

-----------------------------
Dependencies:
- math, numpy, torch, logging, os, sys
- Custom modules: utils

-----------------------------
Usage:
- Run this script as a helper module to generate augmented datasets for all available simulations. 
This module assumes the existence of a 'data/original/' directory containing simulation datasets.
"""

import math
import numpy as np
import torch
import logging
import os
import sys
sys.path.append("")
sys.path.append("..")
from logs import logger
import utils


SOURCE_DIR='data/original/'
SIMULATIONS = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
TESTING_PARAMETERS = {
              "SAVE_LOGS": False,
              "LOGS_FILENAME": "helper_augment_dataset",
              "AMOUNT_SAMPLES": 8,
              "WINDOW_SIZE": [64, 64],
  }

def main():
    logger.logging_config(logs_save=TESTING_PARAMETERS["SAVE_LOGS"], filename=TESTING_PARAMETERS["LOGS_FILENAME"])
    create_augmented_dataset(amount_samples=TESTING_PARAMETERS["AMOUNT_SAMPLES"],window_size=TESTING_PARAMETERS["WINDOW_SIZE"])


def create_augmented_dataset(amount_samples=32,window_size=[64, 64]):
    logging.info(f"[AUGMENT] Augmenting Dataset with Amount Samples: {amount_samples}, Window Size: {window_size}")
    all_datasets_GDM = []
    all_datasets_GSL = []
    for simulation in SIMULATIONS:
        dataset_GDM, dataset_GSL = augment_datasets(simulation, amount_samples=amount_samples, window_size=window_size)
        all_datasets_GDM.append(dataset_GDM)
        all_datasets_GSL.append(dataset_GSL)

    all_datasets_GDM = torch.cat(all_datasets_GDM, dim=1)
    all_datasets_GSL = torch.cat(all_datasets_GSL, dim=1)
    return all_datasets_GDM,all_datasets_GSL


def augment_datasets(dataset=SIMULATIONS[0], amount_samples=32, window_size=None):
    if window_size is None:
        window_size = [64, 64]
    utils.seed_generator()
    dataset_GDM, _ =  utils.load_dataset(dataset)
    all_coordinates = [[x, y] for x in range(49, 139, 15) for y in range(43, 118, 15)]
    all_coordinates = torch.tensor(all_coordinates)
    all_coordinates = all_coordinates.reshape(6, 5, 2)
    all_coordinates = all_coordinates.tolist()


    mapping = create_mapping(all_coordinates=all_coordinates,window_size=window_size)
    stacked_GDM_windows = []
    stacked_GSL = []
    for i in range(len(mapping)):
        y = i % 5
        x = i // 5
        for map in mapping[i]:
            samples = extract_samples(dataset_GDM[i], amount_samples=amount_samples)
            for sample in samples:
                window = extract_window(dataset_GDM[i][sample].unsqueeze(-1), window_size=window_size, source_location=all_coordinates[x][y], new_location=map)
                stacked_GDM_windows.append(window)
            stacked_GSL.append(map)
    dataset_GDM_tensor = torch.stack(stacked_GDM_windows)
    dataset_GDM_tensor = torch.stack(stacked_GDM_windows).reshape(window_size[0]*window_size[1], amount_samples, window_size[0], window_size[1])
    dataset_GSL_tensor = torch.tensor(stacked_GSL).reshape(-1, 2)
    sorted_indices = np.lexsort((dataset_GSL_tensor[:,1].cpu().numpy(), dataset_GSL_tensor[:,0].cpu().numpy()))
    sorted_indices = torch.from_numpy(sorted_indices).long()
    dataset_GDM_tensor = dataset_GDM_tensor[sorted_indices]
    dataset_GSL_tensor = dataset_GSL_tensor[sorted_indices]
    dataset_GSL_tensor = dataset_GSL_tensor.unsqueeze(1).expand(-1, amount_samples, -1)
    return dataset_GDM_tensor, dataset_GSL_tensor

def extract_samples(dataset, amount_samples=32):
    samples = np.random.choice(range(dataset.shape[0]), amount_samples, replace=False)
    return samples

def extract_window(tensor, window_size=(64, 64), source_location=(49, 42), new_location=(0, 0)):
    window = tensor[
        source_location[0] - new_location[0]:source_location[0] - new_location[0] + window_size[0],
        source_location[1] - new_location[1]:source_location[1] - new_location[1] + window_size[1],
        :
    ]
    return window

def create_mapping(all_coordinates,window_size=[64,64]):
    x_ceil = math.ceil(window_size[0]/6)
    y_ceil = math.ceil(window_size[1]/5)
    mapping = []
    for x in range(len(all_coordinates)):
        x_range = range(x_ceil * x, min(x_ceil * (x + 1), window_size[0]))
        for y in range(len(all_coordinates[x])):
            y_range = range(y_ceil * y, min(y_ceil * (y + 1), window_size[1]))
            map = [[i, j] for i in x_range for j in y_range]
            mapping.append(map)
    return mapping

if __name__ == "__main__":
    main()
