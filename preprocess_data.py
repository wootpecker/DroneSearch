import unittest
import numpy as np
import create_dataset
import random
import torch

DATA="train" #train,valid,test                   Old/30x25/
TRANSFORMED=True #reduce 30x25->6x5
SEQUENCE=[2,100]# specify which image to be shown in data of x,y->[x,y,30,25]
SIZE=[6,5]#size of plots   
MAX_ELEMENTS = 24                    #maximal 80% erforscht -> 6x5 -> 24 cells

def seed_generator():                #Random Seed Generator for each function
    SEED=random.seed(16923)

def main():
    dataset=random_cell_generator()
    print(dataset.shape)
    create_dataset.show_as_image_sequence(dataset, SEQUENCE,SIZE)
    return 0

def test():
    dataset = create_dataset.load_combined_dataset(DATA,TRANSFORMED)


def random_cell_generator():
    seed_generator() 
    dataset = create_dataset.load_combined_dataset(DATA,TRANSFORMED)
    array = np.zeros([120, 420, 6, 5])
    shape = dataset.shape
    for i in range(shape[0]):
        find_source_location(dataset,i)
        for j in range(shape[1]):
            amount_of_elements = random.randint(1, MAX_ELEMENTS)
            random_elements = random.sample(range(0, 30), int(amount_of_elements))
            for elements in random_elements:                
                x=int(elements/shape[2])
                y=elements%shape[3]
                array[i,j,x,y]=dataset[i,j,x,y,0]
    return array

def include_source_location(dataset,i):
    #x=random.randint(0,1)
    #if(x>0):
    z=torch.argmax(dataset[i],dim=3)
    y= torch.argmax(z)
    print(z)
    print(y)

def find_source_location(dataset,i):
    shape = dataset.shape
    z=torch.argmax(dataset[i],dim=3)
    y= torch.argmax(z)
    print(z)
    print(y)
    return y


def random_path_generator():
    seed_generator() 
    dataset = create_dataset.load_combined_dataset(DATA,TRANSFORMED)
    




































if __name__ == "__main__":
    main()