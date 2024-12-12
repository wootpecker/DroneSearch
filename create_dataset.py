import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
import utils, datatransformer
from pathlib import Path
from timeit import default_timer as timer 


#from data.gdm_dataset import GasDataSet


DATA="train" #train,valid,test                   Old/30x25/
TRANSFORMED=False #reduce 30x25->6x5
SEQUENCE=[2,100]# specify which image to be shown in data of x,y->[x,y,30,25]   example:39,2
SIZE=[6,5]#size of plots
DATASET_TYPES=["Distinctive","Flattened","S-Shape", "Grid", "Random", "Edge","EncoderDecoder"]
DATASETS=["train","valid","test"]

def main():
    #transform_dataset(DATASET_TYPES[3],distance=3,pad=2)
    #test()
    #test_72x72_all()
    #test_72x72()
    test_24x24_all()
    #create_all_datasets()


def create_all_datasets():
    test=["EncoderDecoder"]
    for dataset in test:
    #for dataset in DATASET_TYPES:
        transform_dataset(dataset,distance=1,pad=0,start_left=True,adequate_input=0)

def test_24x24_all():
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

    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")


def test():
    dataset_GDM,dataset_GSL=utils.load_data(DATASETS[2])
    dataset_GDM = dataset_GDM.reshape(-1, 1, dataset_GDM.shape[-2],dataset_GDM.shape[-1])
    for x in range(dataset_GDM.shape[0]):
        dataset_GDM[x]=datatransformer.transform_datasets_with_type(dataset_GDM=dataset_GDM[x],dataset_type=DATASET_TYPES[3])
        if(x%10000==0):
            print(x)
    print(f"[INFO] result shape: {dataset_GDM.shape}")
    print(f"[INFO] result: {dataset_GDM}") 

def test_72x72():
    test72=torch.rand(100000,1,72,72)
    start_time = timer()
    for x in range(test72.shape[0]):
        test72[x]=datatransformer.transform_single_with_type(dataset_GDM=test72[x],dataset_type=DATASET_TYPES[3])
        if(x%10000==0):
            print(x)
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

def test_72x72_all():
    test72=torch.rand(100000,1,72,72)
    start_time = timer()
    test72=datatransformer.transform_datasets_with_type(dataset_GDM=test72,dataset_type=DATASET_TYPES[3])
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")


def transform_dataset(dataset_type="Flattened",distance=5,pad=1,start_left=True,adequate_input=30):
    target_dir_path = Path(f"data/MyTensor/datasets_{dataset_type}")
    target_dir_path.mkdir(parents=True, exist_ok=True)
    start_time = timer()
    for dataset in DATASETS:
        start_time_data = timer()
        dataset_GDM,dataset_GSL=utils.load_data(dataset)
        dataset_GDM = dataset_GDM.reshape(-1, 1, dataset_GDM.shape[-2],dataset_GDM.shape[-1])
        dataset_GDM_reshaped=dataset_GDM
        if(dataset_type==DATASET_TYPES[0]):   #distinctive
            dataset_GSL=datatransformer.transform_datasets_with_distinctive_source(dataset_GSL)
        else:
            dataset_GDM,dataset_GSL=datatransformer.transform_datasets_with_type(dataset_GDM=dataset_GDM,dataset_GSL=dataset_GSL,dataset_type=dataset_type,distance=distance,pad=pad,start_left=start_left,adequate_input=adequate_input)
        if(dataset_type==DATASET_TYPES[6]): 
            dataset_GSL=dataset_GDM_reshaped
            
        print(f"[INFO] Dataset GSL shape: {dataset_GSL.shape}")
        print(f"[INFO] Dataset GDM shape: {dataset_GDM.shape}")        
        utils.save_dataset(dataset_GDM,dataset_GSL,dataset_type,dataset)
        end_time_data = timer()
        print(f"[INFO] Total Transform time ({dataset}): {end_time_data-start_time_data:.3f} seconds")
    end_time = timer()
    print(f"[INFO] Total Transform time: {end_time-start_time:.3f} seconds")
    test_transform(dataset_type)

def test_transform(dataset_type="Flattened"):    
    
    start_time = timer()
    for dataset in DATASETS:
        target_dir_path = Path(f"data/MyTensor/datasets_{dataset_type}/{dataset}.pt")
        start_time_data = timer()
        x=torch.load(target_dir_path)
        dataset_GSL=x['X']
        dataset_GDM=x['y']
        print(f"[TEST] Dataset GSL shape: {dataset_GSL.shape}")
        print(f"[TEST] Dataset GDM shape: {dataset_GDM.shape}")        
        end_time_data = timer()
        print(f"[INFO] Total Transform time ({dataset}): {end_time_data-start_time_data:.3f} seconds")
    end_time = timer()
    print(f"[INFO] Total Transform time: {end_time-start_time:.3f} seconds")


class SuperDataset(torch.utils.data.Dataset):
    """Dataset with X,y,classes train/test/valid splitted by file"""
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

