from torch.utils.data import Dataset, DataLoader, random_split
import torch
import utils
from timeit import default_timer as timer 


SIMULATIONS = ["01_Winter", "02_Spring", "03_Summer", "04_Autumn"]
MODELS = ["VGG", "EncoderDecoder", "VGGVariation"]
DATASETS=["train","test"]

def main():
    #create_dataloader_distinctive(batch_size=16)
    
    start_time = timer()
    #load_and_split_dataset(simulation)
    create_dataloader(model_type=MODELS[2], batch_size=32) #"test"
    #load_and_split_dataset("test")
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds") 






def create_dataloader(model_type=MODELS[0], batch_size=32):
    train_dataset, test_dataset = load_reshape_dataset(model_type=model_type)
    classes=train_dataset.classes
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"[DATA] Model Type: {model_type}, Batch size: {batch_size}, Classes: {classes}) created.")
    return train_dataloader,test_dataloader, classes

def load_reshape_dataset(model_type=MODELS[0]):
    """
    Loads and reshapes the dataset based on the specified model type.
    Parameters:
    model_type (str): The type of model to use for reshaping the dataset. Defaults to the first model in the MODELS list.
    Returns:
    tuple: A tuple containing the training dataset and the testing dataset, both as instances of SuperDataset.
    The function performs the following steps:
    1. Loads the training and testing datasets using the `utils.load_dataset` function with augmentation enabled.
    2. If the specified model type is the first model in the MODELS list, reshapes the training and testing labels (GSL) to have an additional dimension.
    3. Prints the model type and the shapes of the reshaped training and testing labels.
    4. Creates instances of SuperDataset for the training and testing datasets.
    5. Returns the training and testing datasets as a tuple.
    """

    train_GDM,train_GSL = utils.load_dataset(dataset_name=DATASETS[0], augmented=True)
    test_GDM,test_GSL = utils.load_dataset(dataset_name=DATASETS[1], augmented=True)
    classes=1
    if(model_type==MODELS[0] or model_type==MODELS[2]):
        train_GSL = train_GSL.reshape(train_GSL.shape[0], -1)
        test_GSL = test_GSL.reshape(test_GSL.shape[0], -1)
        classes=None
    print(f"[RESHAPE] Model_Type: {model_type}, train_GSL: {train_GSL.shape}, test_GSL: {test_GSL.shape}")
    #print(f"[RESHAPE] train_GDM: {train_GDM.shape}, test_GDM: {test_GDM.shape}")
    train_dataset = SuperDataset(train_GDM, train_GSL,classes)
    test_dataset = SuperDataset(test_GDM, test_GSL,classes)

    return train_dataset, test_dataset


class SuperDataset(torch.utils.data.Dataset):
    """Dataset with X,y,classes train/test/valid splitted by file"""
    def __init__(self,X, y, classes=None):
        self.X = X
        self.y = y
        if classes is None:
            self.classes=y.shape[-1]
        else:
            self.classes=classes

    def __getitem__(self, index):
        return self.X[index],self.y[index]

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    main()

