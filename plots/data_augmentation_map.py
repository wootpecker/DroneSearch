"""
data_augmentation_map.py
This module provides utility functions for visualizing the data augmentation process.
It includes functions to plot the original and augmented locations of data samples on a grid.

-----------------------------
Testing Parameters:
- LOAD_SEED (int): Random seed for reproducibility.
- SAVE_LOGS (bool): Whether to save or show logs during dataset initialization and processing.
- LOGS_FILENAME (str): Filename for saving logs.

-----------------------------
Functions:
- main(): Configures logging and triggers the plotting of data augmentation map.

- plot_original_locations(): 
    Plots and saves the gas source locations of the dataset provided by Nicolas Winkler on a blank grid.

- plot_augmented(): 
    Plots and saves the augmented sample regions, visualizing the data augmentation.

-----------------------------
Dependencies:
- matplotlib, numpy, pathlib, math, logging
- Custom modules: logs, utils

-----------------------------
Usage:
- Run this script directly or from parent directory to generate and save the original and augmented location plots.
    python data_augmentation_map.py
    DroneSearch directory: python plots\data_augmentation_map.py

- The generated plots are saved in the 'results/data_augmentation/' directory.
"""



import matplotlib.pyplot as plt
import sys
import math
from pathlib import Path
import numpy as np
import logging

sys.path.append("")
sys.path.append("..")
from logs import logger
import utils



TESTING_PARAMETERS = {
              "LOAD_SEED":  16923,
              "SAVE_LOGS": False,
              "LOGS_FILENAME": "data_augmentation_map",
}





def main():
    logger.logging_config(logs_save=TESTING_PARAMETERS["SAVE_LOGS"])
    plot_original_locations()
    plot_augmented()




def plot_original_locations():
    image=np.zeros((180,150,1))    
    plt.figure(figsize=(11, 5))
    all_coordinates = [[x, y] for x in range(49, 139, 15) for y in range(43, 118, 15)]
    target_dir_path = Path("results/data_augmentation/")
    
    target_dir_path.mkdir(parents=True, exist_ok=True)
    plt.imshow(image, origin="lower")
    utils.seed_generator(SEED=TESTING_PARAMETERS['LOAD_SEED'])

    randomizer = np.random.rand(len(all_coordinates), 3)
    for index in range(len(all_coordinates)):
        color = randomizer[index]
        plt.plot(all_coordinates[index][1], all_coordinates[index][0], marker="*", markersize=10, markeredgecolor=color, markerfacecolor=color)
    plt.xlabel("x (dm)", fontsize=18)
    plt.ylabel("y (dm)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(target_dir_path / f'GSL_original.pdf')  
    #plt.savefig(target_dir_path / f'GSL_augmented.pdf')
    logging.info(f"[PLOT] Augmented sample locations saved to {target_dir_path / f'GSL_augmented.pdf'}")

    plt.show()



def plot_augmented():
    utils.seed_generator(SEED=TESTING_PARAMETERS['LOAD_SEED'])
    all_coordinates = [[x, y] for x in range(49, 139, 15) for y in range(43, 118, 15)]
    image = np.zeros((64, 64, 3))    
    plt.figure(figsize=(11, 5))
    x_iterator = math.ceil(64 / 6)
    y_iterator = math.ceil(64 / 5)
    randomizer = np.random.rand(30, 3)
    target_dir_path = Path(f"results/data_augmentation/")
    target_dir_path.mkdir(parents=True, exist_ok=True)
    plt.imshow(image, origin="lower")

    for i in range(6):
        for j in range(5):
            x_start = i * x_iterator
            x_end = min(x_start + x_iterator, 64) 
            y_start = j * y_iterator
            y_end = min(y_start + y_iterator, 64)
            color = randomizer[i * 5 + j]
            image[x_start:x_end, y_start:y_end] = color 
    plt.imshow(image, origin="lower")
    for x in range(0, 64, 1):
        plt.axvline(x=x, color='black', linestyle='--', linewidth=0.3)
    for y in range(0, 64, 1):
        plt.axhline(y=y, color='black', linestyle='--', linewidth=0.3)
    plt.xlabel("x (dm)", fontsize=18)
    plt.ylabel("y (dm)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(target_dir_path / f'GSL_augmented.pdf')
    logging.info(f"[PLOT] Augmented sample locations saved to {target_dir_path / f'GSL_augmented.pdf'}")
    plt.show()











if __name__ == "__main__":
    main()