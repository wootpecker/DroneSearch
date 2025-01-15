import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
from torch.utils import data
from timeit import default_timer as timer 


DATASET_TYPES=["Distinctive","Flattened","S-Shape", "Grid", "Random", "Edge","EncoderDecoder"]
DATASETS=["train","valid","test"]
SIMULATIONS = ["01_Winter", "02_Spring", "03_Summer", "04_Autumn"] 
ALL_SIMULATIONS = "all_simulations"
MODELS = ["VGG", "EncDec"]


def main():
    pass
    #load_and_split_dataset()


def load_and_split_dataset(datasets_GDM,datasets_GSL, train_ratio=0.8):
    """
    Loads the dataset and splits it into train and test sets.
    
    Args:
        data_path (str): Path to the dataset file.
        train_ratio (float): Ratio of the dataset to use for training.
        augmented (bool): Whether to apply augmen++tation.
        
    Returns:
        train_dataset (Dataset): Training dataset.
        test_dataset (Dataset): Testing dataset.
    """
    start_time = timer()
    X = datasets_GDM
    y = datasets_GSL
    #X = data['X']
    #y = data['y']
    X = X.reshape(-1,1,64,64)
    y = y.reshape(-1,2)
    # Calculate the number of training samples
    #print(f"[INFO] X: {X.shape}, y: {y.shape}")

    # Random Seed
    utils.seed_generator()
    
    # Split the dataset
    print(f"[INFO] Dataset loaded and split into train and test sets.")
    train_size = int(train_ratio * X.shape[0])
    indices = torch.randperm(X.shape[0])
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_GDM = X[train_indices, :, :, :]
    test_GDM = X[test_indices, :, :, :]
    
    train_GSL = y[train_indices, :]
    test_GSL = y[test_indices, :]

    
    
    # Create SuperDataset instances
    train_GSL = coordinates_to_grid(train_GSL)
    test_GSL = coordinates_to_grid(test_GSL)


    print(f"[Super] X Train shape: {train_GDM.shape}, y Train shape: {train_GSL.shape}")
    print(f"[Super] X Test shape: {test_GDM.shape}, y Test shape: {test_GSL.shape}")  
    end_time = timer()
    print(f"[INFO] Total Transform time: {end_time-start_time:.3f} seconds")
    #return train_GDM, train_GSL, test_GDM, test_GSL
    utils.save_dataset(train_GDM, train_GSL, "train", augmented=True)
    utils.save_dataset(test_GDM, test_GSL, "test", augmented=True)
    

def coordinates_to_grid(dataset_GSL):
    """
    Converts the coordinates to a grid.
    
    Args:
        y (Tensor): The coordinates.
        
    Returns:
        grid (Tensor): The grid.
    """
    grid = torch.zeros((dataset_GSL.shape[0],1, 64, 64))
    for i in range(dataset_GSL.shape[0]):
        x, y = dataset_GSL[i]
        grid[i,0, x, y] = 1
    return torch.FloatTensor(grid)






def transform_datasets_with_type(dataset_GDM,dataset_GSL,dataset_type,distance=3,pad=1,start_left=True,adequate_input=30):
    #adequate_input=30
    if(dataset_type==DATASET_TYPES[1]):   #flattened input ->30x25->750
        return dataset_GDM
    elif(dataset_type==DATASET_TYPES[2]): #S-Shape source (distance between cross, offset from border)
        coordinates=generate_coordinates_s_shape(dataset_GDM.shape,distance=distance,pad=pad,start_left=start_left)
    else:
        coordinates=generate_coordinates_s_shape(dataset_GDM.shape,distance=1,pad=1,start_left=start_left)
    dataset_GDM=dataset_GDM.squeeze()
    dataset_GDM=do_transformation(dataset_GDM=dataset_GDM,coordinates=coordinates,adequate_input=adequate_input)
    dataset_GSL = dataset_GSL.reshape(-1, dataset_GSL.shape[-1]*dataset_GSL.shape[-2])
    #dataset_GSL = dataset_GSL.reshape(-1,1, dataset_GSL.shape[-2],dataset_GSL.shape[-1])
    return dataset_GDM, dataset_GSL


def transform_single_with_type(dataset_GDM,dataset_type,randomizer=None,distance=3,pad=1,start_left=True,adequate_input=30):
    #adequate_input=30
    if(dataset_type==DATASET_TYPES[2]): #S-Shape source (distance between cross, offset from border)
        coordinates=generate_coordinates_s_shape(dataset_GDM.shape,distance=distance,pad=pad,start_left=start_left,)
    elif(dataset_type==DATASET_TYPES[3]): #Grid (distance between points, offset from border)
        coordinates=generate_coordinates_grid(dataset_GDM.shape,distance=distance,pad=pad)
    elif(dataset_type==DATASET_TYPES[4]): #Random ()
        coordinates=generate_coordinates_random_single(dataset_GDM=dataset_GDM.shape,randomizer=randomizer,distance=distance,pad=pad)
    elif(dataset_type==DATASET_TYPES[5]): #Edge of plume (start fro\\\m source -> find border )
        coordinates=generate_coordinates_grid(dataset_GDM.shape,distance=distance,pad=pad)
    dataset_GDM=dataset_GDM.squeeze()
    dataset_GDM=single_adequate_input(dataset_GDM=dataset_GDM,coordinates=coordinates,adequate_input=adequate_input)
    return dataset_GDM




def single_adequate_input(dataset_GDM,coordinates,adequate_input=30):
    if(adequate_input>0):
        transformed_dataset= np.zeros_like(dataset_GDM)
        for x,y in coordinates:
            #if 0<=x<dataset_GDM.shape[-2] and 0 <= y < dataset_GDM.shape[-1]:
            transformed_dataset[x,y]=dataset_GDM[x,y]
        if(np.sum(transformed_dataset>0.1)>adequate_input):
            dataset_GDM=torch.from_numpy(transformed_dataset)
            #print(np.sum(transformed_dataset>0))
        return dataset_GDM.unsqueeze(0).unsqueeze(0)
    else:
        transformed_dataset= np.zeros_like(dataset_GDM)
        for x,y in coordinates:
                transformed_dataset[x,y]=dataset_GDM[x,y]
        result =torch.from_numpy(transformed_dataset)
        return result.unsqueeze(0).unsqueeze(0)

def do_transformation(dataset_GDM,coordinates,adequate_input=30):
    if(adequate_input>0):
        for i in range(dataset_GDM.shape[0]):
            dataset=dataset_GDM[i]
            transformed_dataset= np.zeros_like(dataset)
            for x,y in coordinates:
                #if 0<=x<dataset_GDM.shape[-2] and 0 <= y < dataset_GDM.shape[-1]:
                transformed_dataset[x,y]=dataset[x,y]
            if(np.sum(transformed_dataset>0.1)>adequate_input):
                dataset_GDM[i]=torch.from_numpy(transformed_dataset)
                #print(np.sum(transformed_dataset>0))
            if(i%10000==0):
                print(f"[INFO] Sample number: {i}")

        return dataset_GDM.unsqueeze(1)
    else:
        transformed_dataset= np.zeros_like(dataset_GDM)
        for x,y in coordinates:
            #if 0<=x<dataset_GDM.shape[-2] and 0 <= y < dataset_GDM.shape[-1]:
                transformed_dataset[:,x,y]=dataset_GDM[:,x,y]
        #transformed_dataset = transformed_dataset.reshape(-1,1, transformed_dataset.shape[-1]*transformed_dataset.shape[-2])
        #result=torch.from_numpy(transformed_dataset)
        return torch.from_numpy(transformed_dataset).unsqueeze(1)


def generate_coordinates_s_shape(dataset_GDM,distance=3,pad=2,start_left=True):
    width,height=dataset_GDM[-2]-1,dataset_GDM[-1]-1
    coordinates = []
    x,y=pad,pad
    while(y<=height-pad):
        #left to right
        if(start_left):
            while(x<=width-pad):
                coordinates.append([x,y])
                x+=1
            x-=1
        else:
            x=width-pad
            while(x>=pad):                
                coordinates.append([x,y])
                x-=1
            x+=1
            
        #x += 1 if not start_left else -1
        y_max = min(y + distance, height - pad)        
        while(y<y_max):
            y+=1
            coordinates.append([x,y])

        start_left= not start_left
        y+=1
    return coordinates

def generate_coordinates_grid(dataset_GDM,distance=3,pad=2):
    width,height=dataset_GDM[-2]-1,dataset_GDM[-1]-1
    distance+=1
    coordinates=[]
    x=pad
    while(x <= width-pad):
        y=pad
        while(y <= height-pad):
            coordinates.append([x,y])
            y += distance
        x += distance
    return coordinates

class CoordinateShuffler:
    def __init__(self, rng):
        self.rng = rng
    def shuffle_coordinates(self, coordinates):
        new_seed= self.rng.integers(0, 2**32)
        new_randomizer = np.random.default_rng(new_seed)
        new_randomizer.shuffle(coordinates)
        return coordinates

def generate_coordinates_random_single(dataset_GDM,randomizer,distance=3,pad=2):
    width,height=dataset_GDM[-2],dataset_GDM[-1]
    reduced_datapoints=(width-2*pad)*(height-2*pad)
    distance=min(distance,9)
    random_datapoints=int(reduced_datapoints*(10-distance)/10)
    all_coordinates = [(x, y) for x in range(pad,width-pad-1) for y in range(pad,height-pad-1)]
    randomizer.shuffle(all_coordinates)
    return all_coordinates[:random_datapoints]

def generate_coordinates_random(shuffler,dataset_GDM,distance=3,pad=2):
    width,height=dataset_GDM[-2],dataset_GDM[-1]
    reduced_datapoints=(width-2*pad)*(height-2*pad)
    distance=min(distance,9)
    random_datapoints=int(reduced_datapoints*(10-distance)/10)
    all_coordinates = [(x, y) for x in range(pad,width-pad-1) for y in range(pad,height-pad-1)]
    shuffled_coordinates= shuffler.shuffle_coordinates(all_coordinates)
    return shuffled_coordinates[:random_datapoints]









if __name__ == "__main__":
    main()
