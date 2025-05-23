"""
data_transformations_with_adequate_input.py
This module provides a collection of data transformation classes and helper functions for augmenting and manipulating grid-like tensor data, 
particularly for use in training machine learning models.
This class has been used for testing purposes, but it is not used in the main code.
Furthermore, it has been used to test the adequacy of the input data for the training of the model.

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

- adequate_input: 
    Creates a threshold for data transformation, ensuring that the transformed data meets a minimum activation criteria.
    Criteria includes a minimum number of activated points in the transformed tensor, that are over a threshold value.
    
-----------------------------
Testing Functions:
- test_noise: Tests the NoiseTransform class by applying noise to a tensor and plotting the result.
- test_sshape_cage: Tests S-shape and cage coordinate generation and transformation.
- test_input: Tests cage coordinate generation and transformation.
- test_dimesions: Tests S-shape coordinate generation for various distances and paddings.
- test_transform: Tests the RandomTransform class on a tensor.

-----------------------------
Dependencies:
- torch, torchvision.transforms, random

-----------------------------
Usage:
- Import and use the transformation classes during training to increase sample variety.
"""



from torchvision import transforms
import torch
import utils
import random



def main():
    #test_dimesions()
    #test_input()
    #test_sshape_cage()
    test_transform()
    #test_noise()

def test_noise():
    log_normalized_data = torch.log1p(torch.rand(1, 64, 64))  # Simulated log-normalized data

    # Create the transform with desired noise level
    noise_transform = NoiseTransform(std=0.05)

    # Apply the noise transform
    noisy_data = noise_transform(log_normalized_data).unsqueeze(0)
    log_normalized_data=log_normalized_data.unsqueeze(0)
    data=torch.stack((log_normalized_data, noisy_data), dim=0)
    utils.plot_more_images(data)
    print("Original Data:", log_normalized_data)
    print("Noisy Data:", noisy_data)

def test_sshape_cage():
    dimensions=[64,64]
    rotate=1
    distance=5
    pad=1
    dataset_GDM=torch.zeros([1,64,64])
    for i in range(0,4096):
        x=int(i/64)
        y=i%64
        dataset_GDM[0, x, y] =1 
        #dataset_GDM[0, int(i/64), i%64] =1 
    coordinates=generate_coordinates_s_shape(dataset_GDM.shape,distance=distance,pad=pad)
    dataset_transformed=adequate_input(dataset_GDM,coordinates)
    if(rotate==1):
        dataset_GDM = transforms.functional.rotate(dataset_GDM, 90)
        #dataset_transformed= transforms.functional.rotate(dataset_transformed, 90)
        coordinates=generate_coordinates_s_shape(dataset_GDM.shape,distance=distance-1,pad=pad+2,start_left=False)
        dataset_transformed2=adequate_input(dataset_GDM,coordinates)
        dataset_transformed2= transforms.functional.rotate(dataset_transformed2, 90)
    
    # Write each element bigger than 0 from dataset_transformed2 to dataset_transformed
        dataset_transformed[dataset_transformed2 > 0] = dataset_transformed2[dataset_transformed2 > 0]
    utils.plot_image(dataset_transformed)
    #print(f"{coordinates}")
    #print(f"{dataset_transformed}")

def test_input():
    dimensions=[64,64]
    distance=5
    pad=1
    dataset_GDM=torch.zeros([1,64,64])
    for i in range(0,4096):
        dataset_GDM[0, int(i/64), i%64] =1 #torch.rand(1) * 0.15

    coordinates=generate_coordinates_cage(dataset_GDM.shape,distance=distance,pad=pad)
    dataset_transformed=adequate_input(dataset_GDM,coordinates)
    utils.plot_image(dataset_transformed)
    print(f"{coordinates}")
    print(f"{dataset_transformed}")

def test_dimesions():
    dimensions=[64,64]
    grids=[]
    sshapes=[]
    distance=10
    pad=10
    all_possibliities = [distance, pad]
    all_possibliities = [(x, y) for x in range(0,distance) for y in range(0,pad)]
    for distance,pad in all_possibliities:
        #grid=generate_coordinates_grid(dimensions,distance=distance,pad=pad)
        sshape =generate_coordinates_s_shape(dimensions,distance=distance,pad=pad)
        print(f"[INFO] s-shape len with distance,pad ({distance},{pad}): {len(sshape)}")

#        X and Y Transform

def test_transform():
    dimensions=[64,64]
    distance=1
    pad=1
    dataset_GDM=torch.zeros([1,64,64])
    for i in range(0,4096):
        x=int(i/64)
        y=i%64
        #if x>30 and x<40:
        #    if y>30 and y<40:
        dataset_GDM[0, x, y] =torch.rand(1) * 0.11 
        #dataset_GDM[0, int(i/64), i%64] =1 
    transformer=RandomTransform()
    dataset_transformed=transformer(x=dataset_GDM,distance=distance,pad=pad)
    #coordinates=generate_coordinates_s_shape(dataset_GDM.shape,distance=distance,pad=pad)
    #dataset_transformed=adequate_input(dataset_GDM,coordinates)
    utils.plot_image(dataset_transformed)
    #print(f"{coordinates}")
    #print(f"{dataset_transformed}")


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
        x_transformed=adequate_input(dataset_GDM=x, coordinates=coordinates,adequate_input=30)
        if len(x_transformed.shape) < 2:
            coordinates=generate_coordinates_s_shape(x.shape,distance=max(distance//2,1),pad=pad,start_left=True)  
            x_transformed=adequate_input(dataset_GDM=x, coordinates=coordinates)
            if len(x_transformed.shape) < 2:
                #dataset_mask=generate_no_mask(x)
                return x#,dataset_mask 
        #dataset_mask=generate_mask(x,coordinates)            
        return x_transformed#,dataset_mask 



class GridTransform:
    def __call__(self, x, distance=1, pad=1):
        coordinates=generate_coordinates_grid(x.shape,distance=distance,pad=pad)        
        x_transformed=adequate_input(dataset_GDM=x, coordinates=coordinates,adequate_input=30)
        if len(x_transformed.shape) < 2:
            coordinates=generate_coordinates_s_shape(x.shape,distance=distance,pad=pad,start_left=True) 
            x_transformed=adequate_input(dataset_GDM=x, coordinates=coordinates)
            if len(x_transformed.shape) < 2:
                #dataset_mask=generate_no_mask(x)
                return x#,dataset_mask 
        #dataset_mask=generate_mask(x,coordinates)            
        return x_transformed#,dataset_mask 
    

    
class CageTransform:
    def __call__(self, x, distance=1, pad=1):
        coordinates= generate_coordinates_cage(x.shape,distance=distance,pad=pad)        
        x_transformed=adequate_input(dataset_GDM=x, coordinates=coordinates,adequate_input=30)
        if len(x_transformed.shape) < 2:
            coordinates=generate_coordinates_s_shape(x.shape,distance=max(distance//2,1),pad=pad,start_left=True)             
            x_transformed=adequate_input(dataset_GDM=x, coordinates=coordinates)
            if len(x_transformed.shape) < 2:
                #dataset_mask=generate_no_mask(x)
                return x#,dataset_mask 
        #dataset_mask=generate_mask(x,coordinates)            
        return x_transformed#,dataset_mask 


class RandomTransform:
    def __call__(self, x, distance=1, pad=1):
        coordinates= generate_coordinates_random(x.shape,distance=distance,pad=pad)        
        x_transformed=adequate_input(dataset_GDM=x, coordinates=coordinates,adequate_input=30)
        if len(x_transformed.shape) < 2:
            coordinates=generate_coordinates_cage(x.shape,distance=distance,pad=pad)
            x_transformed=adequate_input(dataset_GDM=x, coordinates=coordinates)
            if len(x_transformed.shape) < 2:
                #dataset_mask=generate_no_mask(x)
                return x#,dataset_mask 
        #dataset_mask=generate_mask(x,coordinates)            
        return x_transformed#,dataset_mask  









#         Helper Functions

def generate_coordinates_s_shape(dataset_GDM,distance=1,pad=1,start_left=True):
    """
    Generates a list of coordinates in an S-shaped pattern within the given dataset dimensions.
    Args:
        dataset_GDM (list): A list containing the dimensions of the dataset.
        distance (int, optional): The vertical distance of each horizontal pass. Defaults to 3.
        pad (int, optional): The padding to apply to the edges of the dataset. Defaults to 2.
        start_left (bool, optional): Whether to start the pattern from the left side. Defaults to True.
    Returns:
        list: A list of [x, y] coordinates representing the S-shaped pattern.
    """
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
    #print(len(coordinates))
    return coordinates

def generate_coordinates_random(dataset_GDM, distance=3, pad=1):
    #width, height = dataset_GDM.shape[-2]-1, dataset_GDM.shape[-1]-1
    width,height=dataset_GDM[-2],dataset_GDM[-1]
    reduced_datapoints = (width - 2 * pad) * (height - 2 * pad)
    distance = min(distance, 9)
    random_datapoints = int(reduced_datapoints * (10 - distance) / 10)
    all_coordinates = [(x, y) for x in range(pad, width - pad) for y in range(pad, height - pad)]
    random.shuffle(all_coordinates)
    return all_coordinates[:random_datapoints]


def generate_coordinates_cage(dataset_GDM,distance=3,pad=1):
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
    
def generate_coordinates_grid(dataset_GDM,distance=3,pad=1):
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



    
def adequate_input(dataset_GDM,coordinates,adequate_input=20):   
    adequate_input=0
    dataset_GDM = dataset_GDM.squeeze()    
    transformed_dataset = torch.zeros_like(dataset_GDM)
    if adequate_input > 0:
        indices = torch.tensor(coordinates).t()
        transformed_dataset[indices[0], indices[1]] = dataset_GDM[indices[0], indices[1]]
        
        if torch.sum(transformed_dataset > 0.3) > adequate_input:
            return transformed_dataset.unsqueeze(0)
        else:
            return torch.zeros(1)
    else:
        indices = torch.tensor(coordinates).t()
        transformed_dataset[indices[0], indices[1]] = dataset_GDM[indices[0], indices[1]]
        return transformed_dataset.unsqueeze(0)


def generate_mask(dataset_GDM, coordinates):
    """
    Converts the coordinates to a grid with the shape of dataset_GDM.
    
    Args:
        dataset_GDM (Tensor): The dataset tensor.
        coordinates (list): The list of coordinates.
        
    Returns:
        grid (Tensor): The grid tensor with the same shape as dataset_GDM.
    """
    dataset_GDM = dataset_GDM.squeeze() 
    grid = torch.zeros_like(dataset_GDM)
    indices = torch.tensor(coordinates).t()
    grid[indices[0], indices[1]] = 1
    return grid.unsqueeze(0)

def generate_no_mask(dataset_GDM):
    """
    Converts the coordinates to a grid with the shape of dataset_GDM.
    
    Args:
        dataset_GDM (Tensor): The dataset tensor.
        coordinates (list): The list of coordinates.
        
    Returns:
        grid (Tensor): The grid tensor with the same shape as dataset_GDM.
    """
    mask = torch.ones_like(dataset_GDM)
    return mask



if __name__ == "__main__":
    main()
