import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from data.gdm_dataset import GasDataSet

def main():
    dataset=load_data()
    dataset_GDM=dataset["GDM"] 
    print(dataset_GDM)
    print(dataset_GDM.size())   
    dataset_GSL=dataset["GSL"] 
    print(dataset_GSL)
    print(dataset_GSL.size())

    dataset_mixed=combine_datasets(dataset_GDM.numpy(),dataset_GSL.numpy())
    transformed_GSL=transform_into_6x5_all_sequences(dataset_GSL)





def load_data():
    dataset = torch.load("data/BNA_Files/train.pt")
    return dataset


def show_as_image_sequence(dataset, element):
    X = dataset[element]
    f, arr = plt.subplots(3,5) 
    arr[0,0].imshow(X[0].squeeze())
    print('test')
    for i in range(arr.shape[0]*arr.shape[1]):
        iy=int(i%5)
        ix=int(i/5)
        arr[ix,iy].imshow(X[i].squeeze())
        print("i: "+str(i)+"    ix: "+str(ix)+"     iy: "+str(iy))
    plt.show()
    print("x:")
    print(X.size())


def transform_into_6x5_all_sequences(dataset_GSL):
    transformed=[]
    t=0
    for all_sequences in dataset_GSL:
        transformed.append(transform_into_6x5(all_sequences.numpy()))
        print(t)
        t=t+1
    for x in transformed:
        print(x[0])
    return transformed


def transform_into_6x5(dataset_GSL_sequence):
    transform=[6,5]
    result = []
    #print(dataset_GSL_sequence.size())
    for gsl in dataset_GSL_sequence:
        transformed=np.zeros((6,5))
        #gsl=gsl.numpy()
        max=find_max(gsl)
        for coordinates in max:
            transformed[int(coordinates[0]/transform[1]),int(coordinates[1]/transform[1])]=transformed[int(coordinates[0]/transform[1]),int(coordinates[1]/transform[1])]+1
        result.append(transformed)
    return result

def find_max(gsl):
    coordinates=[]
    for x in range(gsl.shape[0]):
        for y in range(gsl.shape[1]):
            if(gsl[x,y]>0):
                coordinates.append([x,y])
    return coordinates

def combine_datasets(dataset_GDM,dataset_GSL):

    dataset_mixed=np.dstack([dataset_GDM,dataset_GSL])
    #print(dataset_mixed)
    #print(dataset_mixed.size())
    dataset_mixed=np.stack((dataset_GDM,dataset_GSL),axis=-1)
    #print(dataset_mixed)
    #print(dataset_mixed.size)
    print(dataset_mixed.dtype)
    print(dataset_mixed.shape)
    #dataset_mixed_temp=np.stack((dataset_mixed,dataset_GSL),axis=-1)
    result=np.zeros((120,420,30,25,3,), dtype='f')
    print(result.dtype)
    print(result.shape)
    for i in range(dataset_mixed.shape[0]):
        print(i)
        for j in range (dataset_mixed.shape[1]):
            for x in range(dataset_mixed.shape[2]):
                for y in range(dataset_mixed.shape[3]):
                    result[i][j][x][y][0]=dataset_GDM[i][j][x][y]
                    result[i][j][x][y][1]=dataset_GSL[i][j][x][y]
    result = torch.tensor(result)
    torch.save(result, "data/BNA_Files/MyTensor/train_combined.pt")
    show_as_image_sequence(result,0)
    return dataset_mixed

def show_dataset(dataset_mixed):
    return dataset_mixed
    














#old code Test Area







def test_area():
    dataset = GasDataSet("data/30x25/test.pt")
    loader = data.DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    data_iter = iter(loader)




    X_Map = torch.load("data/30x25/train.pt")
    X_Real_Map = torch.load("data/6x5_realData/02apr.file")
    print(X_Map.shape[2])
    print(X_Map.size())
    print(X_Real_Map.shape[2])
    print(X_Real_Map.size())


    x_train= []
    y_train =[]


    print("dataset data shape: ")
    print(dataset.data.size())
    for x in dataset.data:
        #print(x.size())
        if(x.size(dim=1)>5):
            x_train.append(x[0])
            #y_train.append(x[0].size())
        else:
            y_train.append(x[1])

    show_as_image_sequence(2000)


    for z in range(5):
        #plt.figure()
        plt.imshow(x_train[z].squeeze())
        f, axarr = plt.subplots(2,1) 
        axarr[0].imshow(x_train[z].squeeze())
        axarr[1].imshow(y_train[z].squeeze())
        f.suptitle('Image Nr. %z'+str(z) )
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('second plot')
        ax1.imshow(x_train[z])
        ax2.imshow(y_train[z])
        plt.show()






if __name__ == "__main__":
    main()
