import math
import numpy as np
import torch
import utils
import logging

processed_dir = 'data/datasets_tensor/'
SIMULATIONS = ["01_Winter", "02_Spring", "03_Summer", "04_Autumn"]
#SIMULATIONS = ["01_Winter"]

def main():
    create_augmented_dataset(amount_samples=8,window_size=[64, 64])
    #test_load(SIMULATIONS[0])

def create_augmented_dataset(amount_samples=32,window_size=[64, 64]):
    """
    Creates an augmented dataset by processing multiple simulations.
    This function iterates over a predefined list of simulations (SIMULATIONS),
    augments the datasets for each simulation using the `augment_datasets` function,
    and concatenates the results. The augmented datasets are then saved using the
    `utils.save_dataset` function.
    Returns:
        None
    """
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
    #utils.save_dataset(all_datasets_GDM, all_datasets_GSL, "all_simulations", augmented=True)
    #return all_datasets_GDM, all_datasets_GSL
    #utils.save_dataset(test_GDM, test_GSL, "test", augmented=True)


def augment_datasets(dataset=SIMULATIONS[0], amount_samples=32, window_size=None):    
    """
    Augments the given dataset by extracting sliding windows and creating new samples.
    Parameters:
    dataset (str): The dataset to be augmented. Defaults to the first element in SIMULATIONS.
    amount_samples (int): The number of samples to extract from each window. Defaults to 32.
    window_size (list or tuple): The size of the sliding window. Defaults to [64, 64] if not provided.
    Returns:
    None: The function saves the augmented dataset to disk and plots some of the augmented images.
    Notes:
    - The function seeds the random generator for reproducibility.
    - It loads the dataset and generates a grid of coordinates for window extraction.
    - The function creates a mapping of coordinates and extracts samples from the dataset.
    - It stacks the extracted windows and sorts them based on their coordinates.

    """
    if window_size is None:
        window_size = [64, 64]
    utils.seed_generator()
    dataset_GDM, _ =  utils.load_dataset(dataset)
    all_coordinates = [[x, y] for x in range(49, 139, 15) for y in range(43, 118, 15)]
    all_coordinates = torch.tensor(all_coordinates)
    all_coordinates = all_coordinates.reshape(6, 5, 2)
    all_coordinates = all_coordinates.tolist()
    #print(f"all_coordinates: {all_coordinates[0]}, \nall_coordinates.shape: {all_coordinates.shape}")

    #test_coordinates = [[x, y] for x in range(0,180) for y in range(0,151)]
    #test_coordinates=torch.tensor(test_coordinates).reshape(180,151,2)
    #print(f"test_coordinates: {test_coordinates[0]}, \ntest_coordinates.shape: {test_coordinates.shape}")

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
    #print(f"window_size: {window.shape()}")
    dataset_GDM_tensor = torch.stack(stacked_GDM_windows)
    dataset_GDM_tensor = torch.stack(stacked_GDM_windows).reshape(window_size[0]*window_size[1], amount_samples, window_size[0], window_size[1])
    dataset_GSL_tensor = torch.tensor(stacked_GSL).reshape(-1, 2)
    # Sort dataset_GDM_tensor and dataset_GSL_tensor based on the first (x) and then the second (y) values of dataset_GSL_tensor
    sorted_indices = np.lexsort((dataset_GSL_tensor[:,1].cpu().numpy(), dataset_GSL_tensor[:,0].cpu().numpy()))
    sorted_indices = torch.from_numpy(sorted_indices).long()
    dataset_GDM_tensor = dataset_GDM_tensor[sorted_indices]
    dataset_GSL_tensor = dataset_GSL_tensor[sorted_indices]
    # Increase the shape of dataset_GSL_tensor to [4096, 32, 2]
    dataset_GSL_tensor = dataset_GSL_tensor.unsqueeze(1).expand(-1, amount_samples, -1)
    #print(f"dataset_GDM_tensor.shape: {dataset_GDM_tensor.shape}")
    #print(f"stacked_GSL_tensor.shape: {dataset_GSL_tensor.shape}")
    #print(f"dataset_GSL_tensor: {dataset_GSL_tensor}")
    #utils.plot_more_images(dataset_GDM_tensor[32*32:32*33,0,:,:], title=dataset)
    #utils.save_dataset(dataset_GDM_tensor, dataset_GSL_tensor, dataset, augmented=True)
    return dataset_GDM_tensor, dataset_GSL_tensor

def extract_samples(dataset, amount_samples=32):
    """
    Extracts a number of random samples from the dataset.
    
    Args:
        dataset (Tensor): The dataset to extract samples from.
        amount_samples (int): The number of samples to extract.
        
    Returns:
        np.ndarray: The indices of the extracted samples.
    """
    #print(f"dataset.shape: {dataset.shape}")
    samples = np.random.choice(range(dataset.shape[0]), amount_samples, replace=False)
    #print(f"samples: {samples}")
    return samples

def extract_window(tensor, window_size=(64, 64), source_location=(49, 42), new_location=(0, 0)):
    """
    Extracts a window from the tensor at the source location to the new location.
    
    Args:
        tensor (Tensor): The input tensor
        window_size (list): The size of the window to extract
        source_location (list): The location of the gas source
        new_location (list): The location in the new window
        
    Returns:
        Tensor: The extracted window
    """

    # Extract the window from the tensor
    window = tensor[
        source_location[0] - new_location[0]:source_location[0] - new_location[0] + window_size[0],
        source_location[1] - new_location[1]:source_location[1] - new_location[1] + window_size[1],
        :
    ]
    return window

def create_mapping(all_coordinates,window_size=[64,64]):
    """
    Create a mapping from the original gas source coordinates to the new coordinates.
    
    Args:
        all_coordinates (list): The list of all coordinates.
        
    Returns:
        np.array: The mapping from the original gas source coordinates to the new coordinates.
    """
    x_ceil = math.ceil(window_size[0]/6)
    y_ceil = math.ceil(window_size[1]/5)
    mapping = []
    for x in range(len(all_coordinates)):
        x_range = range(x_ceil * x, min(x_ceil * (x + 1), window_size[0]))
        for y in range(len(all_coordinates[x])):
            y_range = range(y_ceil * y, min(y_ceil * (y + 1), window_size[1]))
            map = [[i, j] for i in x_range for j in y_range]
            mapping.append(map)
            #print(f"all_coordinates[{x}][{y}]: {all_coordinates[x][y]}")
    return mapping

if __name__ == "__main__":
    main()
