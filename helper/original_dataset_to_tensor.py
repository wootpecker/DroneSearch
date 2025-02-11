import os
import numpy as np
import torch
import utils
#from .. import utils
import logging
from tqdm.auto import tqdm
from tqdm import *

# Define the directory where your simulation files are located
# old: source_dir = 'raw/sim_WS29'

SOURCE_DIR = 'data/original/'


def main():
    # Create dataset tensor with specified parameters
    create_dataset_tensor(log_normalize=True, plume_threshold=10)


def create_dataset_tensor(log_normalize=True, plume_threshold=10):
    """
    Transforms the text files into a Tensor and saves it for each season.
    
    Args:
        log_normalize (bool): Whether to apply logarithmic normalization to the dataset.
        plume_threshold (int): The threshold value for the plume in seconds.
        
    Returns:
        None
    
    Prints:
    """
    directories = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    #print(directories)
    logging.info(f"Folderlist: {directories} (Should be one for each season or wind simulation)")
    #for batch, (X, y) in tqdm(enumerate(test_dataloader), desc="Working", total=len(test_dataloader)):
    with tqdm(directories, position=tqdm._get_free_pos(),leave=False, desc=f'Working on directory: ', total=len(directories)) as folder_range:
    #for folder in tqdm(directories,position=0, desc=f"Working on directory: ", total=len(directories)):                
        for folder in folder_range:
            #count = 0       
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

                #print(filename)
                #if count > 1:
                #     break
                #count += 1

            datasets = np.array(datasets)
            if log_normalize:
                datasets = normalize_dataset(datasets)
            # all_files.append(datasets)
            datasets = torch.FloatTensor(datasets)
            unique_positions = find_max_sequence(datasets)
            utils.save_dataset(datasets, unique_positions, folder)
            # break


def transform_to_dataset(input_file, sample_height=180, plume_threshold=10):
    """
    Transforms a text file into a NumPy array.
    
    Args:
        input_file (str): Path to the input text file.
        sample_height (int): Number of lines per sample (height of each sample).
        
    Returns:
        list: A sequence of samples as a NumPy array.
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Convert lines to a list of 2D-Array (space-separated values)
    data = [list(map(float, line.split())) for line in lines]
    
    # Number of samples
    total_lines = len(data)
    if total_lines % sample_height != 0:
        raise ValueError("[ERROR] The total number of lines is not divisible by the sample height.")
    
    num_samples = total_lines // sample_height
    #plume_threshold=0
    # Group data into a sequence of samples
    dataset = []
    for i in range(plume_threshold * 2, num_samples):
        start = i * sample_height
        end = start + sample_height
        sample = data[start:end]
        dataset.append(np.array(sample))
        #if i <3:
            #print(f"x: {np.array(sample).argmax()//151}, y: {np.array(sample).argmax()%151}")
    
    return np.array(dataset)


def normalize_dataset(data):
    """
    Normalize the dataset.
    
    Args:
        data (np.array): The dataset to normalize.
        
    Returns:
        np.array: The normalized dataset.
    """
    epsilon = 1e-10  # Small constant to avoid log(0)
    log_data = np.maximum(np.log(data + epsilon), 0)
    #log_data = np.maximum(np.log(data), 0)
    return (log_data - np.min(log_data)) / (np.max(log_data) - np.min(log_data))


def find_max_sequence(dataset):
    """
    Find the position of the maximum value of all the sequences in the dataset.
    
    Args:
        dataset (list): A list of samples, where each sample is a Tensor.
        
    Returns:
        list(tensor): The maximum position of each sequence in the dataset as a tensor.
    """
    max_positions = dataset.view(dataset.size(0), dataset.size(1), -1).argmax(dim=2)
    height_indices = max_positions // dataset.size(3)
    width_indices = max_positions % dataset.size(3)

    # Combine height and width positions
    max_positions_2d = torch.stack((height_indices, width_indices), dim=-1)

    # Compare the max positions across sequences for each sample
    unique_count = []
    for sample_idx in range(dataset.size(0)):
        sample_positions = max_positions_2d[sample_idx]  # Positions for this sample
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
