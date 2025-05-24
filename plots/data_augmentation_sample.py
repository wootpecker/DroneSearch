"""
data_augmentation_sample.py
This script visualizes a sample from an augmented dataset to present a data augmentation sample.
It loads a dataset, selects a sample, and plots it with annotations indicating the gas source and the new gas distribution map area after augmentation.

-----------------------------
Testing Parameters:
- LOAD_SEED (int): Random seed for reproducibility.
- SAVE_LOGS (bool): Whether to save or show logs during dataset initialization and processing.
- LOGS_FILENAME (str): Filename for saving logs.

Constants:
- SOURCE_DIR: Directory containing dataset of Nicolas Winkler.
- SIMULATIONS: List of available simulation datasets in the source directory.

-----------------------------
Functions:
- main(): Configures logging and triggers the plotting of an augmented sample.

- plot_augmented_sample(): 
    Loads a dataset sample, applies augmentation, and visualizes it with matplotlib.

-----------------------------
Dependencies:
- matplotlib, numpy, pathlib, math, logging, sys, os
- Custom modules: logs.logger, utils

-----------------------------
Usage:
- Run this script directly or from parent directory to generate and save a plot of an augmented dataset sample.
    python data_augmentation_sample.py
    DroneSearch directory: python plots\data_augmentation_sample.py    

- The generated plots are saved in the 'results/data_augmentation/' directory.

"""

import numpy as np
import sys
import logging
import os
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("")
sys.path.append("..")
from logs import logger
import utils






TESTING_PARAMETERS = {
    "LOAD_SEED": 16923,
    "SAVE_LOGS": False,
    "LOGS_FILENAME": "data_augmentation_sample",
}
SOURCE_DIR='data/original/'
SIMULATIONS = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]



def main():
    logger.logging_config(logs_save=TESTING_PARAMETERS["SAVE_LOGS"], filename=TESTING_PARAMETERS["LOGS_FILENAME"])
    plot_augmented_sample()



def plot_augmented_sample(): 
    logging.info(f"[AUGMENT] Augmenting Dataset for Plotting a Sample")
    dataset =SIMULATIONS[0]   
    dataset_GDM, _ =  utils.load_dataset(dataset)
    all_coordinates = [[x, y] for x in range(49, 139, 15) for y in range(43, 118, 15)]
    sample = 100
    dataset_GDM = dataset_GDM[0][sample].unsqueeze(-1)

    logging.info(f"Shape of plotted GDM dataset: {dataset_GDM.shape}")


    target_dir_path = Path(f"results/data_augmentation/")
    target_dir_path.mkdir(parents=True, exist_ok=True)
    utils.seed_generator(SEED=TESTING_PARAMETERS['LOAD_SEED'])    
    randomizer = np.random.rand(len(all_coordinates), 3)  # Generate random RGB colors
    color = randomizer[0]  # Use the random RGB color
    plt.plot(all_coordinates[0][1], all_coordinates[0][0], marker="*", markersize=10, markeredgecolor=color, markerfacecolor=color, label="Gas Source", linestyle='None')
    plt.imshow(dataset_GDM.numpy(), origin="lower")
    plt.xlabel("x (dm)", fontsize=18)
    plt.ylabel("y (dm)", fontsize=18)
    plt.legend(["Gas Source"], fontsize=14)
    x, y = all_coordinates[0][1], all_coordinates[0][0]
    square_size = 64  
    lower_left_x = x - 10
    lower_left_y = y - 10
    rect = plt.Rectangle(
        (lower_left_x, lower_left_y),
        square_size,
        square_size,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )

    plt.gca().add_patch(rect)

    plt.tight_layout()
    plt.savefig(target_dir_path / f'GSL_augmented_sample.pdf')
    plt.show()
    
    return



if __name__ == "__main__":
    main()
