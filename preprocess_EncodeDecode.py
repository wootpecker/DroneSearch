import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from timeit import default_timer as timer 



DATA="train" #train,valid,test                   Old/30x25/
TRANSFORMED=False #reduce 30x25->6x5
SEQUENCE=[2,100]# specify which image to be shown in data of x,y->[x,y,30,25]   example:39,2



def main():
    test_squares()
    #test_create_24_Dataset()



def test_create_24_Dataset():
    all_coordinates = [(x, y) for x in range(0,24) for y in range(0,24)]        
    print(all_coordinates)
    dataset=[]
    for coordinate in all_coordinates:
        dataset=test_example_test(coordinate,dataset)
    dataset_tensor = torch.FloatTensor(dataset)    
    test24=torch.rand(1000,1,24,24)
    start_time = timer()
    x=torch.load("data/MyTensor/datasets_EncoderDecoder/train.pt")
    x['X']=test24    
    x['y']=test24
    torch.save(x,"data/MyTensor/datasets_EncoderDecoder/train.pt")
    x=torch.load("data/MyTensor/datasets_EncoderDecoder/test.pt")
    x['X']=test24    
    x['y']=test24
    torch.save(x,"data/MyTensor/datasets_EncoderDecoder/test.pt")

def test_example_test(coordinate,dataset):
    for i in range(3):
        example=create_dataset(i,coordinate)
        dataset.append(example)
    return dataset


def create_dataset(i,coordinate):
    zeros=np.zeros([1,24,24])
#    if(i==0):
    

def test_squares():
    # Example usage
    array = create_descending_squares(center=[20,20],array_size=24, max_layers=100)
    #array=create_descending_squares23(center=[2,2])
    print(array)
    plt.imshow(array, cmap="viridis")
    plt.colorbar()
    plt.show()


def create_descending_squares(center=None,array_size=24, max_layers=12):
    """
    Create squares in a 2D array with layers descending from the inner layer to the outer layer.

    Parameters:
        array_size (int): Size of the square array (e.g., 24 for a 24x24 array).
        max_layers (int): Maximum distance (number of layers) from the center to the outermost square.

    Returns:
        numpy.ndarray: A 2D array with descending square layers.
    """
    # Initialize the array
    image = np.zeros((array_size, array_size), dtype="float32")
    
    # Calculate the center of the array
    if center is None:
        center = (array_size // 2, array_size // 2)
    center_x, center_y = center

    for layer in range(max_layers):
        layer=max_layers-layer-1
        if(layer==max_layers-1):
            value=1
        else:
            value = min(np.round(float((max_layers - layer)/max_layers ),decimals=1),0.9)

        start_x = max(center_x - layer,0)
        end_x = min(center_x + layer, array_size - 1)

        start_y = max(center_y - layer, 0)
        end_y = min(center_y + layer, array_size - 1)
        
        # Draw top and bottom rows of the square
        image[start_x, start_y:end_y + 1] = value
        image[end_x, start_y:end_y + 1] = value
        
        # Draw left and right columns of the square
        image[start_x:end_x + 1, start_y] = value
        image[start_x:end_x + 1, end_y] = value
    return image


def create_descending_squares23(array_size=24, max_layers=12, center=None):
    array = np.zeros((array_size, array_size), dtype=int)
    
    # Set the center
    if center is None:
        center = (array_size // 2, array_size // 2)
    center_x, center_y = center
    
    # Generate layers
    for layer in range(max_layers):
        #layer=max_layers-layer-1
        value = max_layers - layer  # Descending value for each layer
        
        # Calculate bounds for the square
        start_x = max(center_x - layer, 0)
        end_x = min(center_x + layer, array_size - 1)
        start_y = max(center_y - layer, 0)
        end_y = min(center_y + layer, array_size - 1)
        
        # Apply values to the edges of the square
        for x in range(start_x, end_x + 1):
            array[x, start_y] = value  # Left edge
            array[x, end_y] = value    # Right edge
        for y in range(start_y, end_y + 1):
            array[start_x, y] = value  # Top edge
            array[end_x, y] = value    # Bottom edge
    
    return array

if __name__ == "__main__":
    main()


