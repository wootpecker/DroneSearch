"""
logger.py

This module provides logging configuration utilities for the DroneSearch project.
It allows for flexible logging setup, including saving logs to files with customizable filenames
based on training parameters, or simply logging to the console.

Dependencies:
- logging, os, pathlib, math, logging

Functions:
- logging_config(logs_save=True, amount_samples=4, transform=True, model_type="EncoderDecoder", window_size=[64,64], filename=None):
    Configures the logging system. If logs_save is True, logs are saved to a file in the logs/files directory,
    with the filename reflecting the training configuration. Otherwise, logs are output to the console.

- main():
    Example entry point that initializes logging with logs_save disabled for testing purposes.

Usage:
    Import and call logging_config() at the start of your script to set up logging as needed.

"""

import logging
import os
from pathlib import Path
 

def main():
    logging_config(logs_save=False)
    
def logging_config(logs_save=True, amount_samples=4, transform=True, model_type="EncoderDecoder", window_size=[64,64],filename=None):
    if logs_save:
        target_dir_path = Path(f"logs")
        target_dir_path.mkdir(parents=True, exist_ok=True)
        target_dir_path = Path(f"logs/files")
        target_dir_path.mkdir(parents=True, exist_ok=True)
        files=os.listdir(target_dir_path)
        if transform:
            transform="TR"
        else:
            transform="NO"
        if filename==None:
            fn=f"logs/files/training_{len(files)+1:03d}_{window_size[0]}x{window_size[1]}_{amount_samples}_{model_type}_{transform}.log"
        else:
            fn=f"logs/files/{filename}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s -- [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M::%S",
            filename=fn,
            #filename=f"logs/test.log",
            filemode='w'
            )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s -- [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M::%S"        
            )    

    logging.info("Logging Configuration Loaded.")



if __name__ == "__main__":
    main()
