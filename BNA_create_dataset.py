import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
#from data.gdm_dataset import GasDataSet





def main():
    data="train" #train,valid,test                   Old/30x25/
    transformed=False #reduce 30x25->6x5
    sequence=[39,100]# specify which image to be shown in data of x,y->[x,y,30,25]
    size=[6,5]#size of plots
    dataset_GDM,dataset_GSL=load_data(data)
    load_imgshow_dataset(sequence,data,transformed,size)#show image sequence 39
    #load_imgshow_dataset(39,"train_combined_6x5.pt")
    #save datasets [GDM,GSL] -> RG image
    #dataset_mixed=combine_datasets(dataset_GDM.numpy(),dataset_GSL.numpy(),"train_combined.pt")

    #save datasets [GDM,GSL,0] -> RGB image
    #save_imgshow_dataset(dataset_mixed,"train_combined_imshow.pt")#save image sequences as tensor
    

    #transform to 6x5 matrix
    transform_datasets(transformed,dataset_GDM,dataset_GSL)


#load data
def load_data(name):
    dataset = torch.load("data/"+name+".pt")
    dataset_GDM=dataset["GDM"]    
    dataset_GSL=dataset["GSL"] 
    return dataset_GDM,dataset_GSL


#load and show dataset in plots of size
def load_imgshow_dataset(sequence,name,transformed,size):    
    if(transformed):
        name = name + "_combined_6x5_imgshow"
    else:
        name = name + "_combined_imgshow"
    dataset = torch.load("data/MyTensor/"+name+".pt") 
    show_as_image_sequence(dataset,sequence,size)
    
def show_as_image_sequence(dataset, sequence,size):
    X = dataset[sequence[0]]
    f, arr = plt.subplots(size[0],size[1]) 
    for i in range(arr.shape[0]*arr.shape[1]):
        iy=int(i%size[1])
        ix=int(i/size[1])
        arr[ix,iy].imshow(X[i+sequence[1]].squeeze())
    plt.show()



def transform_datasets(transformed,dataset_GDM,dataset_GSL):
    if(transformed):
        dataset_GSL=transform_gsl_6x5(dataset_GSL)
        dataset_GDM=transform_gdm_6x5(dataset_GDM)
        #transformed_GSL=find_distinctive_source(dataset_GSL)
        combine_save_dataset(dataset_GDM,dataset_GSL,data+"_combined_6x5")
    else:
        combine_save_dataset(dataset_GDM,dataset_GSL,data+"_combined")


def combine_save_dataset(dataset_GDM,dataset_GSL,name):
    dataset_mixed=combine_datasets(dataset_GDM,dataset_GSL,name)
    save_imgshow_dataset(dataset_mixed,name)

def combine_datasets(dataset_GDM,dataset_GSL,name):
    dataset_mixed = torch.stack((dataset_GDM, dataset_GSL), dim=-1)
    #dataset_mixed=np.stack((dataset_GDM,dataset_GSL),axis=-1)
    #dataset_mixed=torch.tensor(dataset_mixed)
    torch.save(dataset_mixed, "data/MyTensor/"+name+".pt")
    print("saved " +name+"\nshape: ")
    print(dataset_mixed.shape)
    return dataset_mixed

def save_imgshow_dataset(dataset_mixed,name):
    name = name+"_imgshow"
    result =torch.nn.functional.pad(dataset_mixed, (0, 1))
    torch.save(result, "data/MyTensor/"+name+".pt")
    print("saved " +name+"\nshape: ")
    print(result.shape)





def transform_gdm_6x5(dataset_GDM):
    old_shape = dataset_GDM.shape
    new_shape =[6,5]
    # Resizing from 30,25 to 6,5
    row = old_shape[-2] // new_shape[0]
    col = old_shape[-1] // new_shape[1]

    # Perform Slicing of dataset
    reduced_gdm = dataset_GDM[:, :, ::row, ::col]

    return reduced_gdm




def transform_gsl_6x5(dataset_GSL):
    old_shape = dataset_GSL.shape
    new_shape=[6,5]
    # Resizing from 30,25 to 6,5
    row_block_size = old_shape[-2] // new_shape[0] 
    col_block_size = old_shape[-1] // new_shape[1]

    reduced_gsl = torch.zeros((old_shape[0], old_shape[1], new_shape[0], new_shape[1]))
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            block = dataset_GSL[:, :, i*row_block_size:(i+1)*row_block_size, j*col_block_size:(j+1)*col_block_size]
            reduced_gsl[:, :, i, j] = torch.sum(block * (block > 0), dim=(-2, -1))  # Summing over the last two dimensions
    return reduced_gsl


def find_distinctive_source(dataset_GSL_image):
    dataset_GSL_image=dataset_GSL_image.numpy()
    coordinates=[]
    exists=False
    for k in range (dataset_GSL_image.shape[0]):
        for j in range (dataset_GSL_image.shape[1]):
            for x in range(dataset_GSL_image.shape[2]):
                for y in range(dataset_GSL_image.shape[3]):
                    if(dataset_GSL_image[k,j,x,y]>0):
                        if coordinates==[]:
                            coordinates.append([x,y])
                        for z in coordinates:
                            if z ==[x,y]:
                                exists=True
                        if(exists==True):
                            exists=False
                        else:
                            coordinates.append([x,y])
                            print("x: "+str(x)+"y: "+str(y))
    print(len(coordinates))
    return coordinates

if __name__ == "__main__":
    main()

