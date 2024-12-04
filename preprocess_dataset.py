import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
#from data.gdm_dataset import GasDataSet


DATA="train" #train,valid,test                   Old/30x25/
TRANSFORMED=False #reduce 30x25->6x5
SEQUENCE=[2,100]# specify which image to be shown in data of x,y->[x,y,30,25]   example:39,2
SIZE=[6,5]#size of plots


def main():
    #load_imgshow_dataset(DATA,TRANSFORMED,SEQUENCE,SIZE)
    #find_distinctive_source(dataset_GSL_image=dataset_GSL)
    #transform_datasets_with_distinctive_source()
    transform_datasets_flattened()
    #save datasets [GDM,GSL] -> RG image
    #dataset_mixed=combine_datasets(dataset_GDM.numpy(),dataset_GSL.numpy(),"train_combined.pt")

    #transform to 6x5 matrix
    #transform_datasets(TRANSFORMED,DATA,dataset_GDM,dataset_GSL)
    
    #create all datasets
    #create_all_datasets()



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
    datasets=["train","valid","test"]
    for dataset in datasets:
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
    datasets=["train","valid","test"]
    for dataset in datasets:
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









#find all 24,24 possible source locations






if __name__ == "__main__":
    main()

