import os
import torch
import model_dataloader, utils, engine, model_builder,engine_encdec
from timeit import default_timer as timer 
from torchvision import transforms
import numpy
import random


# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LOAD_SEED=16923
TRAIN_SEED=42

MODEL_TYPES = ["VGG", "EncoderDecoder", "VGGVariation"]
DATASET_TYPES = ["01_Winter", "02_Spring", "03_Summer", "04_Autumn"] 


# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    
    results=utils.load_loss(MODEL_TYPES[0])
    for x in results:
        print(f"{x},{len(results[x])}")

    utils.plot_loss_curves(results)
    
    
    #train_all_models(dataloader_type=DATASET_TYPES[0],model_type= MODEL_TYPES[1])  



if __name__ == "__main__":
    main()

