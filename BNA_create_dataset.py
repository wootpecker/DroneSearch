import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from data.gdm_dataset import GasDataSet





def main():
    dataset_GDM,dataset_GSL=load_data()
    load_imgshow_dataset(1,"train_combined_6x5_imgshow")#show image sequence 39
    #load_imgshow_dataset(39,"train_combined_6x5.pt")
    #save datasets [GDM,GSL] -> RG image
    #dataset_mixed=combine_datasets(dataset_GDM.numpy(),dataset_GSL.numpy(),"train_combined.pt")

    #save datasets [GDM,GSL,0] -> RGB image
    #save_imgshow_dataset(dataset_mixed,"train_combined_imshow.pt")#save image sequences as tensor
    
    #transform to 6x5 matrix
    transformed_GSL=transform_gsl_6x5_all_sequences(dataset_GSL)
    #transformed_GSL=reduce_large_tensor_and_sum_by_block(dataset_GSL)
    transformed_GDM=transform_gdm_6x5_all_sequences(dataset_GDM)
    #combine_imgshow_dataset(transformed_GDM,transformed_GSL,"train_combined_6x5")
    

def combine_imgshow_dataset(dataset_GDM,dataset_GSL,name):
    dataset_mixed=combine_datasets(dataset_GDM.numpy(),dataset_GSL,name)
    save_imgshow_dataset(dataset_mixed,name)


def load_data():
    dataset = torch.load("data/BNA_Files/train.pt")
    dataset_GDM=dataset["GDM"]    
    dataset_GSL=dataset["GSL"] 
    return dataset_GDM,dataset_GSL



def combine_datasets(dataset_GDM,dataset_GSL,name):
    dataset_mixed=np.stack((dataset_GDM,dataset_GSL),axis=-1)
    torch.save(dataset_mixed, "data/BNA_Files/MyTensor/"+name+".pt")
    return dataset_mixed


def show_as_image_sequence(dataset, element,size):
    X = dataset[element]
    f, arr = plt.subplots(size[0],size[1]) 
    for i in range(arr.shape[0]*arr.shape[1]):
        iy=int(i%size[1])
        ix=int(i/size[1])
        arr[ix,iy].imshow(X[i+100].squeeze())
    plt.show()

def save_imgshow_dataset(dataset_mixed,name):
    result=np.zeros((dataset_mixed.shape[0],dataset_mixed.shape[1],dataset_mixed.shape[2],dataset_mixed.shape[3],3), dtype='f')
    for i in range(dataset_mixed.shape[0]):
        print(str(i) + "/120")
        for j in range (dataset_mixed.shape[1]):
            for x in range(dataset_mixed.shape[2]):
                for y in range(dataset_mixed.shape[3]):
                    result[i][j][x][y][0]=dataset_mixed[i][j][x][y][0]
                    result[i][j][x][y][1]=dataset_mixed[i][j][x][y][1]
    result = torch.tensor(result)
    torch.save(result, "data/BNA_Files/MyTensor/"+name+"_imgshow.pt")
    print("saved\nshape: ")
    print(result.shape)

def load_imgshow_dataset(number,name):    
    result = torch.load("data/BNA_Files/MyTensor/"+name+".pt")
    show_as_image_sequence(result,number,[6,5])
    





def transform_gsl_6x5_all_sequences(dataset_GSL):
    transformed=[]
    t=0
    for all_sequences in dataset_GSL:
        transformed.append(transform_gsl_6x5(all_sequences.numpy()))
        print(t)
        t=t+1
    return transformed


def transform_gsl_6x5(dataset_GSL_sequence):
    transform=[6,5]
    result = []
    #print(dataset_GSL_sequence.size())
    for dataset_GSL_image in dataset_GSL_sequence:
        transformed=np.zeros((6,5))
        #gsl=gsl.numpy()
        max=find_max(dataset_GSL_image)
        for coordinates in max:
            transformed[int(coordinates[0]/transform[1]),int(coordinates[1]/transform[1])]=transformed[int(coordinates[0]/transform[1]),int(coordinates[1]/transform[1])]+1
        result.append(transformed)
    return result

def find_max(dataset_GSL_image):
    coordinates=[]
    for x in range(dataset_GSL_image.shape[0]):
        for y in range(dataset_GSL_image.shape[1]):
            if(dataset_GSL_image[x,y]>0):
                coordinates.append([x,y])
    return coordinates




def transform_gdm_6x5_all_sequences(dataset_GDM):
    old_shape = dataset_GDM.shape
    new_shape =[6,5]
    # Resizing from 30,25 to 6,5
    row = old_shape[-2] // new_shape[0]
    col = old_shape[-1] // new_shape[1]

    # Perform Slicing of dataset
    reduced_gdm = dataset_GDM[:, :, ::row, ::col]

    return reduced_gdm







# Example usage
def test():
    arr = np.random.rand(120, 420, 30, 25)  # Creating a random array of shape (120, 420, 30, 25)
    reduced_arr = transform_gdm_6x5_all_sequences(arr)
    print(reduced_arr.shape)  # Should print (120, 420, 6, 5)arr = np.random.rand(30, 25)  # Creating a random 30x25 array



def reduce_large_tensor_and_sum_by_block(tensor):
    old_shape = tensor.shape
    new_shape=[6,5]

    row_block_size = old_shape[-2] // new_shape[0]  # 30 -> 6, so 30//6 = 5
    col_block_size = old_shape[-1] // new_shape[1]  # 25 -> 5, so 25//5 = 5

    reduced_tensor = torch.zeros((old_shape[0], old_shape[1], new_shape[0], new_shape[1]))
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            block = tensor[:, :, i*row_block_size:(i+1)*row_block_size, j*col_block_size:(j+1)*col_block_size]
            reduced_tensor[:, :, i, j] = torch.sum(block * (block > 0), dim=(-2, -1))  # Summing over the last two dimensions
    return reduced_tensor




if __name__ == "__main__":
    main()

