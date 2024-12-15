import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
import random
import datatransformer
from torch.utils import data
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer 



DATASET_TYPES=["Distinctive","Flattened","S-Shape", "Grid", "Random", "Edge","EncoderDecoder"]



def main():
    #make_plots()
    test_create_24_Dataset()
    #test_squares()
    #test_create_24_Dataset()



def test_create_24_Dataset(examples=10):
    all_coordinates = [(x, y) for x in range(0,24) for y in range(0,24)]        
    print(all_coordinates)
    dataset_X=[]
    dataset_y=[]
    for coordinate in all_coordinates:
        all_possibilities = [(max_layers, droupout) for max_layers in range(3,17) for droupout in range(40,90)]
        random.shuffle(all_possibilities)
        for i in range(examples):
            array_drop=torch.FloatTensor(create_squares_dropout(center=coordinate,array_size=24, max_layers=all_possibilities[i][0],droupout=all_possibilities[i][1])).unsqueeze(0)
            #array_drop=create_squares_dropout(center=coordinate,array_size=24, max_layers=all_possibilities[i][0],droupout=all_possibilities[i][1])
            dataset_X.append(array_drop)
            dataset_y.append(create_source_location(coordinate))
    dataset_X=torch.stack(dataset_X)
    dataset_y=torch.stack(dataset_y)
    X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_size=0.2, random_state=42)

    train=torch.load("data/MyTensor/datasets_EncoderDecoder/train.pt")
    train['X']=X_train    
    train['y']=y_train
    torch.save(train,"data/MyTensor/datasets_EncoderDecoder/train.pt")
    print(f"[INFO] Train Dataset Saved X: {X_train.shape}")
    print(f"[INFO] Train Dataset Saved y: {y_train.shape}")

    test=torch.load("data/MyTensor/datasets_EncoderDecoder/test.pt")
    test['X']=X_test    
    test['y']=y_test
    torch.save(test,"data/MyTensor/datasets_EncoderDecoder/test.pt")
    print(f"[INFO] Test Dataset Saved X: {X_test.shape}")
    print(f"[INFO] Test Dataset Saved y: {y_test.shape}")


def create_source_location(center,array_size=24):
    image = np.zeros((array_size, array_size), dtype="float32")
    image[center[0],center[1]]=1
    return torch.FloatTensor(image).unsqueeze(0)


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
        if(layer==0):
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


def create_squares_dropout(center=None,array_size=24, max_layers=12,droupout=20):
    """
    Create squares in a 2D array with layers descending from the inner layer to the outer layer.

    Parameters:
        array_size (int): Size of the square array (e.g., 24 for a 24x24 array).
        max_layers (int): Maximum distance (number of layers) from the center to the outermost square.

    Returns:
        numpy.ndarray: A 2D array with descending square layers.
    """
    # Initialize the array
    utils.seed_generator()
    image = np.zeros((array_size, array_size), dtype="float32")
    
    # Calculate the center of the array
    if center is None:
        center = (array_size // 2, array_size // 2)
    center_x, center_y = center
    #older_value=0
    old_value,older_value,oldest_value=0,0,0
    for layer in range(max_layers):
        layer=max_layers-layer-1
        if(layer==0):
            value,old_value,older_value,oldest_value=1,1,1,1
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
        
        
        
        # Apply values to the edges of the square
        for x in range(start_x, end_x + 1):
            result=decide_value(value,old_value,older_value,oldest_value,droupout)
            image[x, start_y] = result  # Left edge
            image[x, end_y] = result    # Right edge
        for y in range(start_y, end_y + 1):
            result=decide_value(value,old_value,older_value,oldest_value,droupout)
            image[start_x, y] = result  # Top edge
            image[end_x, y] = result    # Bottom edge
        oldest_value=older_value
        older_value=old_value
        old_value=value
    return image


def decide_value(value,old_value,older_value,oldest_value,droupout):
    maximum=100
    max_dropout=droupout
    dropout_value=random.randint(0,maximum)
    if(dropout_value<int(max_dropout/4)):
        result=oldest_value
    elif(dropout_value<int(6*max_dropout/10)):
        result=older_value
    elif(dropout_value<max_dropout):
        result= old_value
    else:
        result= value
    return result


def make_plots(array_list=torch.rand([100,1,24,24])):
    array=torch.FloatTensor(create_squares_dropout(center=[3,3],array_size=24, max_layers=10,droupout=10))
    array_drop=torch.FloatTensor(create_squares_dropout(center=[3,3],array_size=24, max_layers=10,droupout=90))
    array_transformed=datatransformer.transform_single_with_type(array,DATASET_TYPES[2],randomizer=None,distance=1,pad=0,start_left=True,adequate_input=0)
    array_drop_transformed=datatransformer.transform_single_with_type(array_drop,DATASET_TYPES[2],randomizer=None,distance=1,pad=0,start_left=True,adequate_input=0)
    size=[1,4]
    f, arr = plt.subplots(size[0],size[1]) 
    random_samples=random.sample(range(array_list.shape[0]),k=5)
    for j in range(arr.shape[0]):
        i=random_samples[j]
        print(i)
        arr[0].imshow(array.numpy())
        arr[0].set(xlabel=f"Sample: {i}, array")
        arr[1].imshow(array_drop.squeeze(0).unsqueeze(-1).numpy())
        arr[1].set(xlabel=f"Sample: {i}, array_drop")
        arr[2].imshow(array_transformed.squeeze().unsqueeze(-1).numpy())
        arr[2].set(xlabel=f"Sample: {i}, array_transformed")
        arr[3].imshow(array_drop_transformed.squeeze().unsqueeze(-1).numpy())
        arr[3].set(xlabel=f"Sample: {i}, y_preds_percent")
        #plt.xticks([])
        #plt.yticks([])
   # f.tight_layout()
    plt.subplots_adjust(hspace =0.4)
    plt.show()




if __name__ == "__main__":
    main()


