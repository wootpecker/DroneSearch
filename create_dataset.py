"""
create_dataset.py
This module provides functionality to initialize, augment, and split datasets for training machine learning models.
It handles the creation of tensor datasets from the 180x150 data generated by N. Winkler, applies augmentation, and splits the data into training and testing sets. 
Logging is supported for tracking dataset initialization and processing steps.

-----------------------------
Testing Parameters:
- SAVE_LOGS (bool): Whether to save or show logs during dataset initialization and processing.
- LOGS_FILENAME (str): Filename for saving logs.
- SAVE_DATASET (bool): Whether to save the processed dataset.
- AMOUNT_SAMPLES (int): Number of samples to generate for each gas source location during augmentation.
- WINDOW_SIZE (list of int): Size of the window to use for data augmentation.
- TRAIN_RATIO (float): Ratio of data to use for training (rest is for testing).

Constants:
- SOURCE_DIR (str): Directory containing the original dataset of 180x150 images.  

-----------------------------
Functions:
- main():
    Used mainly for testing purposes.

- create_dataset(amount_samples, window_size, train_ratio, save_dataset):
    Initializes the dataset directory structure, creates tensor datasets if not present,
    applies augmentation, and splits the data into training and testing sets.
    Returns the split datasets.

- initialize():
    Ensures required directories exist and creates tensor datasets from the original data and saves them, if they do not already exist.

- transformation(amount_samples, window_size, train_ratio, save_dataset):
    Applies data augmentation and splits the datasets into training and testing sets.
    Returns the split datasets.

-----------------------------    
Dependencies:
- pathlib,os,logging
- Custom modules: helper.original_dataset_to_tensor, helper.augment_dataset, helper.train_test_split, logs.logger

-----------------------------
Usage:
- Run this script directly to initialize and process the dataset:
    python create_dataset.py
"""

from helper import original_dataset_to_tensor,augment_dataset, train_test_split
from  logs import logger
from pathlib import Path
import os
import logging

SOURCE_DIR = 'data/original/' # Directory containing the 180x150 dataset

TESTING_PARAMETERS = {
              "SAVE_LOGS": False,
              "LOGS_FILENAME": "initialize_dataset",
              "SAVE_DATASET": False,
              "AMOUNT_SAMPLES": 8,
              "WINDOW_SIZE": [64, 64],
              "TRAIN_RATIO": 0.8
  }


def main():
    logger.logging_config(logs_save=TESTING_PARAMETERS["SAVE_LOGS"],filename=TESTING_PARAMETERS["LOGS_FILENAME"])
    create_dataset()
    
    
def create_dataset(amount_samples=TESTING_PARAMETERS["AMOUNT_SAMPLES"],window_size=TESTING_PARAMETERS["WINDOW_SIZE"],train_ratio=TESTING_PARAMETERS["TRAIN_RATIO"],save_dataset=TESTING_PARAMETERS["SAVE_DATASET"]):
  initialize()
  train_GDM,train_GSL,test_GDM,test_GSL=transformation(amount_samples=amount_samples,window_size=window_size,train_ratio=train_ratio, save_dataset=save_dataset)  
  return train_GDM,train_GSL,test_GDM,test_GSL


def initialize():
  target_dir_path = Path(f"data")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"data/datasets_tensor")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  files=os.listdir(target_dir_path)
  if len(files)==0:
    logging.info("INITIALIZING DATASET")
    original_dataset_to_tensor.create_dataset_tensor(log_normalize=True, plume_threshold=10)

def transformation(amount_samples=TESTING_PARAMETERS["AMOUNT_SAMPLES"],window_size=TESTING_PARAMETERS["WINDOW_SIZE"],train_ratio=TESTING_PARAMETERS["TRAIN_RATIO"],save_dataset=TESTING_PARAMETERS["SAVE_DATASET"]):
   all_datasets_GDM,all_datasets_GSL=augment_dataset.create_augmented_dataset(amount_samples=amount_samples,window_size=window_size)
   train_GDM,train_GSL,test_GDM,test_GSL=train_test_split.load_and_split_dataset(datasets_GDM=all_datasets_GDM,datasets_GSL=all_datasets_GSL,train_ratio=train_ratio,save=save_dataset,window_size=window_size) 
   return train_GDM,train_GSL,test_GDM,test_GSL

if __name__ == "__main__":
    main()
