import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
#from data.gdm_dataset import GasDataSet


DATA="train" #train,valid,test                   Old/30x25/
TRANSFORMED=False #reduce 30x25->6x5
SEQUENCE=[2,100]# specify which image to be shown in data of x,y->[x,y,30,25]   example:39,2
SIZE=[6,5]#size of plots
DATASET_TYPES=["Flattened","Distinctive","S-Shape", "Grid", "Random", "Edge"]
DATASETS=["train","valid","test"]

def main():
    dataset_type=["Flattened","Distinctive","S-Shape", "Grid", "Random", "Edge"]
    #load_imgshow_dataset(DATA,TRANSFORMED,SEQUENCE,SIZE)
    #find_distinctive_source(dataset_GSL_image=dataset_GSL)
    #transform_datasets_with_distinctive_source()
    transform_dataset("Flattened")
    #transform_datasets_flattened()
    #save datasets [GDM,GSL] -> RG image
    #dataset_mixed=combine_datasets(dataset_GDM.numpy(),dataset_GSL.numpy(),"train_combined.pt")

    #transform to 6x5 matrix
    #transform_datasets(TRANSFORMED,DATA,dataset_GDM,dataset_GSL)
    
    #create all datasets
    #create_all_datasets()

def transform_dataset(dataset_type="Flattened"):
    if(dataset_type==DATASET_TYPES[0]):
        transform_datasets_flattened()



def create_all_datasets():
    datasets=["train","valid","test"]
    for dataset in datasets:
        dataset_GDM,dataset_GSL=load_data(dataset)
        transform_datasets(True,dataset,dataset_GDM,dataset_GSL)
        transform_datasets(False,dataset,dataset_GDM,dataset_GSL)

#load data
def load_data(name):
    """
    Load Dataset out of name of file(train,valid,test)
    
    Parameters:
    name(string) : String with train,valid,test to load their dataset
    
    Returns:
    Dataset of GDM(Gas Distribution Map) and GSL(Gas Source Location) seperately
    """
    dataset = torch.load("data/original/"+name+".pt")
    dataset_GDM=dataset["GDM"]    
    dataset_GSL=dataset["GSL"] 
    return dataset_GDM,dataset_GSL

#load GDM and GSL combined
def load_combined_dataset(name,transformed):
    """
    Load combined Dataset out of name of file(train,valid,test)
    
    Parameters:
    name(string) : String with train,valid,test to load their dataset
    transformed(boolean) : If transformed is True, the 6x5 reduced dataset will be loaded

    Returns:
    Dataset of GDM(Gas Distribution Map) and GSL(Gas Source Location) combined in one tensor
    """    
    if(transformed):
        name = name + "_combined_6x5"
    else:
        name = name + "_combined"
    dataset = torch.load("data/MyTensor/simple_dataset/"+name+".pt") 
    return dataset

#load and show dataset in plots of size
def load_imgshow_dataset(name,transformed,sequence,size):    
    if(transformed):
        name = name + "_combined_6x5_imgshow"
    else:
        name = name + "_combined_imgshow"
    dataset = torch.load("data/MyTensor/simple_dataset/"+name+".pt") 
    show_as_image_sequence(dataset,sequence,size)
    return dataset
    
def show_as_image_sequence(dataset, sequence,size):
    """
    Show Dataset as images -> sequence is which wind map we want to use
    
    Parameters:
    dataset(string) : String with train,valid,test to load their dataset
    sequence(boolean) : If transformed is True, the 6x5 reduced dataset will be loaded
    size()
    
    Plots of Datasets 
    """  
    X = dataset[sequence[0]]
    f, arr = plt.subplots(size[0],size[1]) 
    for i in range(arr.shape[0]*arr.shape[1]):
        iy=int(i%size[1])
        ix=int(i/size[1])
        arr[ix,iy].imshow(X[i+sequence[1]].squeeze())
        plt.subplot(size[0],size[1],i+1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(str(sequence[1]+i)+" s")
    plt.show()


def transform_datasets(transformed, data_name, dataset_GDM,dataset_GSL):
    if(transformed):
        dataset_GSL=transform_gsl_6x5(dataset_GSL)
        dataset_GDM=transform_gdm_6x5(dataset_GDM)
        #transformed_GSL=find_distinctive_source(dataset_GSL)
        combine_save_dataset(dataset_GDM,dataset_GSL,data_name+"_combined_6x5")
    else:
        combine_save_dataset(dataset_GDM,dataset_GSL,data_name+"_combined")


def combine_save_dataset(dataset_GDM,dataset_GSL,name):
    dataset_mixed=combine_datasets(dataset_GDM,dataset_GSL,name)
    save_imgshow_dataset(dataset_mixed,name)

def combine_datasets(dataset_GDM,dataset_GSL,name):
    dataset_mixed = torch.stack((dataset_GSL,dataset_GDM), dim=-1)
    #dataset_mixed=np.stack((dataset_GDM,dataset_GSL),axis=-1)
    #dataset_mixed=torch.tensor(dataset_mixed)
    torch.save(dataset_mixed, "data/MyTensor/simple_dataset/"+name+".pt")
    print(f'saved: {name}, shape: {dataset_mixed.shape}, dtype: {dataset_mixed.dtype}')
    return dataset_mixed

def save_imgshow_dataset(dataset_mixed,name):
    #save datasets [GDM,GSL,0] -> RGB image
    #save_imgshow_dataset(dataset_mixed,"train_combined_imshow.pt")#save image sequences as tensor
    name = name+"_imgshow"
    result =torch.nn.functional.pad(dataset_mixed, (0, 1))
    torch.save(result, "data/MyTensor/simple_dataset/"+name+".pt")
    print(f'saved: {name}, shape: {result.shape}, dtype: {result.dtype}')





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
        print(k)
        if k>35:
            break
        for j in range (dataset_GSL_image.shape[1]):
            for x in range(dataset_GSL_image.shape[2]):
                for y in range(dataset_GSL_image.shape[3]):
                    if(dataset_GSL_image[k,j,x,y]>0):
                        if coordinates==[]:
                            coordinates.append([x,y])
                        for z in coordinates:
                            if z ==[x,y]:
                                exists=True
                                break
                        if(exists==True):
                            exists=False
                        else:
                            coordinates.append([x,y])
                            print("x: "+str(x)+"    y: "+str(y))
    print(len(coordinates))
    return coordinates

def transform_datasets_with_distinctive_source():
    for dataset in DATASETS:
        dataset_GDM,dataset_GSL=load_data(dataset)
        coordinates=find_distinctive_source(dataset_GSL)
        coordinates=torch.IntTensor(coordinates)
        x,y=coordinates[:,0],coordinates[:,1]
        dataset_GSL=dataset_GSL[:,:,x,y]
        dataset_GSL = dataset_GSL.reshape(-1, 30)
        dataset_GDM = dataset_GDM.reshape(-1, 1, 30, 25)
        print(dataset_GSL.shape)
        print(dataset_GDM.shape)
        torch.save({'X': dataset_GDM, 'y':dataset_GSL},f"data/MyTensor/datasets_distinctive/{dataset}.pt")


def transform_datasets_flattened():
    for dataset in DATASETS:
        dataset_GDM,dataset_GSL=load_data(dataset)
        dataset_GSL = dataset_GSL.reshape(-1, 750)
        dataset_GDM = dataset_GDM.reshape(-1, 1, 30, 25)
        print(dataset_GSL.shape)
        print(dataset_GDM.shape)
        torch.save({'X': dataset_GDM, 'y':dataset_GSL},f"data/MyTensor/datasets_flattened/{dataset}.pt")



class Combined_Distinctive_Source(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self,data_path):
        data = torch.load(data_path)
        self.X = data['X']
        self.y = data['y']
        self.classes=data['y'].shape[-1]

    def __getitem__(self, index):
        return self.X[index],self.y[index]

    def __len__(self):
        return len(self.y)

if __name__ == "__main__":
    main()

