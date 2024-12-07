import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
from torch.utils import data


DATA="train" #train,valid,test                   Old/30x25/
TRANSFORMED=False #reduce 30x25->6x5
SEQUENCE=[2,100]# specify which image to be shown in data of x,y->[x,y,30,25]   example:39,2
SIZE=[6,5]#size of plots
DATASET_TYPES=["Distinctive","Flattened","S-Shape", "Grid", "Random", "Edge"]
DATASETS=["train","valid","test"]
LOAD_SEED=16923
TRAIN_SEED=42

def main():
    dimensions=[30,25]
    distance=8
    pad=2
    grid=generate_coordinates_grid(dimensions,distance=distance,pad=pad)
    shape=generate_coordinates_s_shape(dimensions,distance=distance,pad=pad)
    randomize = np.random.default_rng(TRAIN_SEED)
    shuffler = CoordinateShuffler(randomize)
    random=generate_coordinates_random(dataset_GDM=dimensions,distance=distance,pad=pad,shuffler=shuffler)
    print(f"[INFO] grid shape: {len(grid)}")
    print(f"[INFO] grid: {grid}") 
    print(f"[INFO] s-shape shape: {len(shape)}")
    print(f"[INFO] s-shape: {shape}") 
    print(f"[INFO] random shape: {len(random)}")
    print(f"[INFO] random: {random}") 
    random=generate_coordinates_random(dataset_GDM=dimensions,distance=distance,pad=pad,shuffler=shuffler)
    print(f"[INFO] random shape: {len(random)}")
    print(f"[INFO] random: {random}") 
    random=generate_coordinates_random(dataset_GDM=dimensions,distance=distance,pad=pad,shuffler=shuffler)
    print(f"[INFO] random shape: {len(random)}")
    print(f"[INFO] random: {random}") 
    random=generate_coordinates_random(dataset_GDM=dimensions,distance=distance,pad=pad,shuffler=shuffler)
    print(f"[INFO] random shape: {len(random)}")
    print(f"[INFO] random: {random}") 

    #S-Shape source (distance between cross, offset from border)
    #Grid (distance between points, offset from border)
    #Random ()
    #Edge of plume (start fro\\\m source -> find border )



def transform_datasets_with_type(dataset_GDM,dataset_type,distance=3,pad=1,start_left=True,adequate_input=30):
    #adequate_input=30
    if(dataset_type==DATASET_TYPES[1]):   #flattened input ->30x25->750
        return dataset_GDM
    elif(dataset_type==DATASET_TYPES[2]): #S-Shape source (distance between cross, offset from border)
        coordinates=generate_coordinates_s_shape(dataset_GDM.shape,distance=distance,pad=pad,start_left=start_left)
    elif(dataset_type==DATASET_TYPES[3]): #Grid (distance between points, offset from border)
        coordinates=generate_coordinates_grid(dataset_GDM.shape,distance=distance,pad=pad)
        adequate_input=10
    elif(dataset_type==DATASET_TYPES[4]): #Random ()
        randomize = np.random.default_rng(TRAIN_SEED)
        shuffler = CoordinateShuffler(randomize)
        coordinates=generate_coordinates_random(dataset_GDM=dataset_GDM.shape,distance=distance,pad=pad,shuffler=shuffler)
    elif(dataset_type==DATASET_TYPES[5]): #Edge of plume (start fro\\\m source -> find border )
        coordinates=generate_coordinates_grid(dataset_GDM.shape,distance=distance,pad=pad)
    dataset_GDM=dataset_GDM.squeeze()
    if(len(dataset_GDM.shape)>2):
        dataset_GDM=do_transformation(dataset_GDM=dataset_GDM,coordinates=coordinates,adequate_input=adequate_input)
    else:
        dataset_GDM=do_single_transformation(dataset_GDM=dataset_GDM,coordinates=coordinates,adequate_input=adequate_input)
    return dataset_GDM

def do_single_transformation(dataset_GDM,coordinates,adequate_input=30):
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
                print(i)
        return dataset_GDM.unsqueeze(1)
    else:
        transformed_dataset= np.zeros_like(dataset_GDM)
        for x,y in coordinates:
            #if 0<=x<dataset_GDM.shape[-2] and 0 <= y < dataset_GDM.shape[-1]:
                transformed_dataset[:,x,y]=dataset_GDM[:,x,y]
        #transformed_dataset = transformed_dataset.reshape(-1,1, transformed_dataset.shape[-1]*transformed_dataset.shape[-2])
        #result=torch.from_numpy(transformed_dataset)
        return torch.from_numpy(transformed_dataset)


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

def generate_coordinates_random2(dataset_GDM,distance=3,pad=2):
    width,height=dataset_GDM[-2],dataset_GDM[-1]
    reduced_datapoints=(width-2*pad)*(height-2*pad)
    distance=min(distance,9)
    random_datapoints=int(reduced_datapoints*(10-distance)/10)
    all_coordinates = [(x, y) for x in range(pad,width-pad-1) for y in range(pad,height-pad-1)]
    randomizer= np.random.default_rng(LOAD_SEED)
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
    print(f"[INFO] Anzahl an GSL: {len(coordinates)}")
    print(f"[INFO] GSL vor Sortierung: {coordinates}")
    coordinates.sort(key=lambda coordinates: coordinates[0])
    print(f"[INFO] GSL nach Sortierung: {coordinates}")
    return coordinates

def transform_datasets_with_distinctive_source(dataset_GSL):
    coordinates=find_distinctive_source(dataset_GSL)
    coordinates=torch.IntTensor(coordinates)
    x,y=coordinates[:,0],coordinates[:,1]
    dataset_GSL=dataset_GSL[:,:,x,y]
    dataset_GSL = dataset_GSL.reshape(-1, dataset_GSL.shape[-1])
    return dataset_GSL



if __name__ == "__main__":
    main()
