import unittest
import numpy as np
import create_dataset
import random
import torch

DATA="train" #train,valid,test                   Old/30x25/
TRANSFORMED=True #reduce 30x25->6x5
SEQUENCE=[2,100]# specify which image to be shown in data of x,y->[x,y,30,25]
SIZE=[6,5]#size of plots   
MAX_ELEMENTS = 30                    #maximal 80% erforscht -> 6x5 -> 24 cells
MIN_ELEMENTS = 29
def seed_generator():                #Random Seed Generator for each function
    SEED=random.seed(16923)

def main():
    dataset=save_preprocessed_data()
    #dataset=random_cell_generator()
    print(dataset.shape)
    create_dataset.show_as_image_sequence(dataset, SEQUENCE,SIZE)
    

def test():
    dataset = create_dataset.load_combined_dataset(DATA,TRANSFORMED)


def load_preprocessed_data():
    return torch.load("data/MyTensor/"+DATA+"_preprocessed.pt") 


def save_preprocessed_data():
    dataset = random_cell_generator(DATA,TRANSFORMED)
    dataset = torch.tensor(dataset)
    name = DATA + "_preprocessed"
    torch.save(dataset, "data/MyTensor/"+name+".pt")
    print(f"saved  {name} \nshape: {dataset.shape}")
    return dataset

def random_cell_generator(data,transformed):
    seed_generator() 
    dataset = create_dataset.load_combined_dataset(data,transformed)
    
    shape = dataset.shape
    shape=np.array(shape)
    if shape[4]>1:
        shape[4]=shape[4]-1
    array = np.zeros(shape)
    for i in range(shape[0]):       
        cells_included_base = np.arange(30)
        for j in range(shape[1]):
            include_source=random.randint(0,1)
            if include_source:
                position=find_single_source_location(dataset[i])
                cells_included = np.delete(cells_included_base, position)
                random_elements = [position]
            else:
                cells_included=cells_included_base
                random_elements=[]

            amount_of_elements = random.randint(MIN_ELEMENTS, MAX_ELEMENTS-include_source)
            random_elements.extend(random.sample(list(cells_included), amount_of_elements))
            
            for element in random_elements:                
                x = element // shape[2]
                y = element % shape[3]
                array[i,j,x,y,0]=dataset[i,j,x,y,1]
    return array

def find_single_source_location(dataset):
    shape = dataset.shape 
    source_locations = dataset[0,..., 0] 
    y=torch.argmax(source_locations)
    #y=torch.argmax(z[0])  
    #print(z[0])
    #print(y)
    xelem=int(y/shape[2])
    yelem=y%shape[2]
    #print(f"corresponding Dataset value{dataset[i,0,xelem,yelem]}")
    if(dataset[0,xelem,yelem,0]!=1):
        print(f"error!!!!! \n  corresponding Dataset value: {dataset[0,xelem,yelem,1]} \n corresponding Dataset {dataset[0]}")
    #source_locations = dataset[..., 1]  # Shape becomes [420, 6, 5]
    return int(y)


def random_path_generator():
    seed_generator() 
    dataset = create_dataset.load_combined_dataset(DATA,TRANSFORMED)
    




































if __name__ == "__main__":
    main()