"""
data_transformations.py
This module provides a collection of data transformation classes and helper functions for augmenting and manipulating grid-like tensor data, 
particularly for use in training machine learning models.

-----------------------------
Classes:
- RotationTransform: Rotates both input and target tensors by a specified angle.
- NoiseTransform: Applies random additive or multiplicative Gaussian noise to a tensor.
- SshapeTransform: Applies an S-shaped sampling pattern to the input tensor.
- GridTransform: Applies a grid-based sampling pattern to the input tensor.
- CageTransform: Applies a cage-like sampling pattern to the input tensor.
- RandomTransform: Applies a random sampling pattern to the input tensor.

-----------------------------
Helper Functions:
- generate_coordinates_s_shape: 
    Generates coordinates in an S-shaped pattern.

- generate_coordinates_cage: 
    Generates coordinates in a cage-like pattern.

- generate_coordinates_grid: 
    Generates coordinates in a grid pattern.

- transform_input: 
    Applies a coordinate-based mask to the input tensor, transforming the grid with the generated patterns.

- generate_coordinates_random: 
    Generates random coordinates within the grid. Creating random activations has been ultimately discarded.    

- generate_mask: 
    Creates a binary mask tensor from coordinates. Creating masks has been ultimately discarded.

- generate_no_mask: 
    Creates a mask tensor with all ones. Creating masks has been ultimately discarded.

-----------------------------
Dependencies:
- torch, torchvision.transforms, random

-----------------------------
Usage:
- Import and use the transformation classes during training to increase sample variety.
"""


import torch
from torchvision import transforms
import random



def main():
    pass


class RotationTransform:
    def __init__(self, rotation_angle=90):
        self.rotation_angle = rotation_angle
    def __call__(self, x, y):
        x = transforms.functional.rotate(x, self.rotation_angle)
        y = transforms.functional.rotate(y, self.rotation_angle)
        return x, y    
    


#        X Transform
class NoiseTransform:
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, x):
        if(torch.rand(1).item()>0.5):
            # Additive noise            
            noise = torch.normal(mean=0.0, std=self.std, size=x.shape, device=x.device)
            noisy_tensor = x + noise
        else:
            # Multiplicative noise            
            noise = torch.normal(mean=1.0, std=self.std, size=x.shape, device=x.device)
            noisy_tensor = x * noise

        # clamp values to a valid range
        noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)

        return noisy_tensor



class SshapeTransform:
    def __call__(self, x, distance=1, pad=1, start_left=True):
        coordinates=generate_coordinates_s_shape(x.shape,distance=distance,pad=pad,start_left=start_left)        
        x_transformed=transform_input(dataset_GDM=x, coordinates=coordinates)        
        return x_transformed 



class GridTransform:
    def __call__(self, x, distance=1, pad=1):
        coordinates=generate_coordinates_grid(x.shape,distance=distance,pad=pad)        
        x_transformed=transform_input(dataset_GDM=x, coordinates=coordinates)    
        return x_transformed
    

    
class CageTransform:
    def __call__(self, x, distance=1, pad=1):
        coordinates= generate_coordinates_cage(x.shape,distance=distance,pad=pad)        
        x_transformed=transform_input(dataset_GDM=x, coordinates=coordinates)         
        return x_transformed 


class RandomTransform:
    def __call__(self, x, distance=1, pad=1):
        coordinates= generate_coordinates_random(x.shape,distance=distance,pad=pad)        
        x_transformed=transform_input(dataset_GDM=x, coordinates=coordinates)
        return x_transformed




#         Helper Functions
def generate_coordinates_s_shape(dataset_GDM,distance=1,pad=1,start_left=True):
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
        y_max = min(y + distance, height - pad)        
        while(y<y_max):
            y+=1
            coordinates.append([x,y])

        start_left= not start_left
        y+=1
    return coordinates

def generate_coordinates_random(dataset_GDM, distance=3, pad=2):
    width,height=dataset_GDM[-2],dataset_GDM[-1]
    reduced_datapoints = (width - 2 * pad) * (height - 2 * pad)
    distance = min(distance, 9)
    random_datapoints = int(reduced_datapoints * (10 - distance) / 10)
    all_coordinates = [(x, y) for x in range(pad, width - pad) for y in range(pad, height - pad)]
    random.shuffle(all_coordinates)
    return all_coordinates[:random_datapoints]


def generate_coordinates_cage(dataset_GDM,distance=3,pad=2):
    width,height=dataset_GDM[-2]-1,dataset_GDM[-1]-1
    coordinates = []
    x,y=pad,pad
    while(y<=height-pad):  
        while(x<=width-pad):
            coordinates.append([x,y])
            x+=1
        x-=1                            
        y_max = min(y + distance, height - pad)        
        while(y<y_max):
            y+=1
            temp_x=pad
            while(temp_x<=width-pad):
                coordinates.append([temp_x,y])
                temp_x+=distance+1
        y+=1
        x=pad
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


    
def transform_input(dataset_GDM,coordinates):   
    dataset_GDM = dataset_GDM.squeeze()    
    transformed_dataset = torch.zeros_like(dataset_GDM)
    indices = torch.tensor(coordinates).t()
    transformed_dataset[indices[0], indices[1]] = dataset_GDM[indices[0], indices[1]]
    return transformed_dataset.unsqueeze(0)


def generate_mask(dataset_GDM, coordinates):
    dataset_GDM = dataset_GDM.squeeze() 
    grid = torch.zeros_like(dataset_GDM)
    indices = torch.tensor(coordinates).t()
    grid[indices[0], indices[1]] = 1
    return grid.unsqueeze(0)

def generate_no_mask(dataset_GDM):
    mask = torch.ones_like(dataset_GDM)
    return mask


if __name__ == "__main__":
    main()
