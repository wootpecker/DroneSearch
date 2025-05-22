"""
Utility functions to make predictions.

Main reference for code creation: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set 
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
#import ..model_dataloader as model_dataloader
from logs import logger
import model_dataloader
import model_builder
import utils
from tqdm.auto import tqdm
import torchmetrics
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import random
import pandas as pd
import math
import logging
import ds4_train_model
from pathlib import Path
import os
import numpy as np
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPES = ["VGG8", "UnetS", "VGGVariation"]
TRANSFORMED_MODEL=True
HYPER_PARAMETERS = ds4_train_model.HYPER_PARAMETERS
TRAINING_PARAMETERS = ds4_train_model.TRAINING_PARAMETERS



HYPER_PARAMETERS['AMOUNT_SAMPLES'] = 1
HYPER_PARAMETERS['TRANSFORM'] = True



MODEL_TO_TEST=[HYPER_PARAMETERS['MODEL_TYPES'][0],HYPER_PARAMETERS['MODEL_TYPES'][1]]
#MODEL_TO_TEST=[HYPER_PARAMETERS['MODEL_TYPES'][1]]


def main():
    
    logger.logging_config(logs_save=False)
    plot_original_locations()
    plot_augmented()




def plot_original_locations():
    image=np.zeros((180,150,1))    
    plt.figure(figsize=(11, 5))
    all_coordinates = [[x, y] for x in range(49, 139, 15) for y in range(43, 118, 15)]
    target_dir_path = Path(f"results/data_augmentation/")
    target_dir_path.mkdir(parents=True, exist_ok=True)
    plt.imshow(image, origin="lower")
    utils.seed_generator(SEED=TRAINING_PARAMETERS['LOAD_SEED'])

    randomizer = np.random.rand(len(all_coordinates), 3)  # Generate random RGB colors
    for index in range(len(all_coordinates)):
        color = randomizer[index]  # Use the random RGB color
        plt.plot(all_coordinates[index][1], all_coordinates[index][0], marker="*", markersize=10, markeredgecolor=color, markerfacecolor=color)
    plt.xlabel("x (dm)", fontsize=18)
    plt.ylabel("y (dm)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(target_dir_path / f'GSL_original.png')
        
    #plt.show()



def plot_augmented():
    utils.seed_generator(SEED=TRAINING_PARAMETERS['LOAD_SEED'])
    all_coordinates = [[x, y] for x in range(49, 139, 15) for y in range(43, 118, 15)]

    image = np.zeros((64, 64, 3))    
    plt.figure(figsize=(11, 5))
    x_iterator = math.ceil(64 / 6)
    y_iterator = math.ceil(64 / 5)
    randomizer = np.random.rand(30, 3)  # Generate random RGB colors for 30 parts
    target_dir_path = Path(f"results/data_augmentation/")
    target_dir_path.mkdir(parents=True, exist_ok=True)
    plt.imshow(image, origin="lower")

    for i in range(6):  # 6 parts along the x-axis
        for j in range(5):  # 5 parts along the y-axis
            x_start = i * x_iterator
            x_end = min(x_start + x_iterator, 64)  # Ensure it doesn't exceed the image boundary
            y_start = j * y_iterator
            y_end = min(y_start + y_iterator, 64)  # Ensure it doesn't exceed the image boundary
            color = randomizer[i * 5 + j]  # Get the color for the current part
            image[x_start:x_end, y_start:y_end] = color  # Assign the color to the corresponding pixels
    plt.imshow(image, origin="lower")
    for x in range(0, 64, 1):
        plt.axvline(x=x, color='black', linestyle='--', linewidth=0.3)
    for y in range(0, 64, 1):
        plt.axhline(y=y, color='black', linestyle='--', linewidth=0.3)
    #plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel("x (dm)", fontsize=18)
    plt.ylabel("y (dm)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(target_dir_path / f'GSL_augmented.png')
    plt.show()











if __name__ == "__main__":
    main()