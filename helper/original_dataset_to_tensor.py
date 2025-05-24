"""
original_dataset_to_tensor.py
This script processes raw text-based dataset of Nicolas Winkler, converting them into normalized PyTorch tensors for further training of machine learning models. 
It is designed to handle multiple directories, which each representing a different season, apply optional logarithmic normalization, and identifies key features such as the most frequent maximum value positions in the data.

-----------------------------
Testing Parameters:
- SAVE_LOGS (bool): Whether to save or show logs during dataset initialization and processing.
- LOGS_FILENAME (str): Filename for saving logs.
- LOG_NORMALIZE (bool): Enables logarithmic normalization of the dataset.
- PLUME_THRESHOLD (int): The threshold value for the plume generation in seconds, used to start measuring when plumes are fully developed.

Constants:
- SOURCE_DIR: Directory containing dataset of Nicolas Winkler.

-----------------------------
Functions:
- main(): Used for testing purposes.

- create_dataset_tensor(log_normalize, plume_threshold):
    Iterates through each dataset directory, processes all text files into tensors, applies normalization if specified, and saves the resulting tensors and positions.

- transform_to_dataset(input_file, sample_height, plume_threshold):
    Reads a text file, splits it into samples of a specified height, and returns a sequence of samples as a NumPy array, 
    skipping initial samples based on the plume threshold.

- normalize_dataset(data):
    Applies logarithmic normalization to the dataset, scaling values between 0 and 1.

- find_max_sequence(dataset):
    For each sample in the dataset, finds the position of the maximum value in each sequence, 
    determines the most frequent maximum positions, and returns them as a tensor for testing purposes.    

-----------------------------
Dependencies:
- numpy, torch, tqdm, logging, sys, os 
- Custom modules: utils, logs.logger

-----------------------------
Usage:
- Run this script as a helper module to generate a tensor file for the dataset of Nicolas Winkler. 
- Run this script directly to process all datasets in the specified SOURCE_DIR.

This module assumes the existence of a 'data/original/' directory containing simulation datasets.
"""



import os
import numpy as np
import torch
import logging
from tqdm.auto import tqdm
from tqdm import *
import sys
sys.path.append("")
sys.path.append("..")
import utils
from logs import logger

SOURCE_DIR = 'data/original/'

TESTING_PARAMETERS = {
              "SAVE_LOGS": True,
              "LOGS_FILENAME": "dataset_to_tensor",    
              "LOG_NORMALIZE": True,
              "PLUME_THRESHOLD": 10,
  }


def main():
    logger.logging_config(logs_save=TESTING_PARAMETERS["SAVE_LOGS"], filename=TESTING_PARAMETERS["LOGS_FILENAME"])
    create_dataset_tensor(log_normalize=TESTING_PARAMETERS["LOG_NORMALIZE"], plume_threshold=TESTING_PARAMETERS["PLUME_THRESHOLD"])


def create_dataset_tensor(log_normalize=True, plume_threshold=10):
    directories = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    logging.info(f"Folderlist: {directories} (Should be one for each season or wind simulation)")
    with tqdm(directories, position=tqdm._get_free_pos(),leave=False, desc=f'Working on directory: ', total=len(directories)) as folder_range:
        for folder in folder_range:
            logging.info(f"Foldername: {folder}")
            path_to_folder = os.path.join(SOURCE_DIR, folder)
            datasets = []
            files=os.listdir(path_to_folder)
            filename=None
            for filename in tqdm(files,position=tqdm._get_free_pos(),leave=False, desc=f"Working on files: ", total=len(files)): 
                filepath = os.path.join(path_to_folder, filename)
                dataset = transform_to_dataset(filepath, sample_height=180, plume_threshold=plume_threshold)
                datasets.append(dataset)
                logging.info(f"Filename: {filename}")

            datasets = np.array(datasets)
            if log_normalize:
                datasets = normalize_dataset(datasets)
            datasets = torch.FloatTensor(datasets)
            unique_positions = find_max_sequence(datasets)
            utils.save_dataset(datasets, unique_positions, folder)


def transform_to_dataset(input_file, sample_height=180, plume_threshold=10):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Convert lines to a list of 2D-Array (space-separated values)
    data = [list(map(float, line.split())) for line in lines]
    
    # Number of samples
    total_lines = len(data)
    if total_lines % sample_height != 0:
        raise ValueError("[ERROR] The total number of lines is not divisible by the sample height.")
    
    num_samples = total_lines // sample_height

    # Group data into a sequence of samples
    dataset = []
    for i in range(plume_threshold * 2, num_samples):
        start = i * sample_height
        end = start + sample_height
        sample = data[start:end]
        dataset.append(np.array(sample))
    
    return np.array(dataset)


def normalize_dataset(data):
    epsilon = 1e-10  # Small constant to avoid log(0)
    log_data = np.maximum(np.log(data + epsilon), 0)
    return (log_data - np.min(log_data)) / (np.max(log_data) - np.min(log_data))


def find_max_sequence(dataset):
    max_positions = dataset.view(dataset.size(0), dataset.size(1), -1).argmax(dim=2)
    height_indices = max_positions // dataset.size(3)
    width_indices = max_positions % dataset.size(3)
    max_positions_2d = torch.stack((height_indices, width_indices), dim=-1)
    unique_count = []
    for sample_idx in range(dataset.size(0)):
        sample_positions = max_positions_2d[sample_idx]  
        unique_positions, counts = torch.unique(sample_positions, dim=0, return_counts=True)  # Unique positions and their counts
        sorted_indices = torch.argsort(counts, descending=True)
        sorted_positions = unique_positions[sorted_indices]
        sorted_counts = counts[sorted_indices]
        logging.info(f"Sample: {sample_idx}")
        for position, count in zip(sorted_positions[:3].tolist(), sorted_counts[:3].tolist()):            
            logging.info(f"Position {position} appears {count} times as the max.")
        unique_count.append(sorted_positions[0].tolist())
    unique_counts = torch.tensor(np.array(unique_count))
    return unique_counts


if __name__ == "__main__":
    main()
