from torch.utils.data import Dataset, DataLoader, random_split
import torch
import utils
from timeit import default_timer as timer 
from torchvision import transforms
import data_transformations

SIMULATIONS = ["01_Winter", "02_Spring", "03_Summer", "04_Autumn"]
MODELS = ["VGG", "EncoderDecoder", "VGGVariation"]
DATASETS=["train","test"]

def main():
    #create_dataloader_distinctive(batch_size=16)
    #test()
    plot_with_classes(model_type=MODELS[1])

 

def plot_with_classes(model_type=MODELS[1]):
    utils.seed_generator()
    train_dataset, test_dataset= load_reshape_dataset(model_type=model_type)
    a=0
    b=a+15
    single_x=[]
    single_y=[]
    for i in range(a,b):
        print(f"{i}:")
        temp=train_dataset.__getitem__(i)
        max_value = torch.max(temp[0])
        print(f"Max value in the dataset: {max_value}")
        single_y.append(temp[1])
        single_x.append(temp[0])
    single_x = torch.stack(single_x)
    single_y = torch.stack(single_y)
    print(f"Max value in the dataset: {max_value}")
    #z=train_dataset[a:b]

    
    single_z = []
    print("new dataset test:")
    train_dataset, test_dataset= load_reshape_dataset(model_type=model_type,transform=False)
    z=train_dataset[a:b]
    x=z[0]
    y=z[1]
    z = []
    true_z=train_dataset[a:b]
    true_x=true_z[0]
    true_y=true_z[1]
    true_z=[]
    for i in range(0, x.shape[0], 5):
        z.extend(x[i:i+5])
        z.extend(y[i:i+5])
        single_z.extend(single_x[i:i+5])
        single_z.extend(single_y[i:i+5])
        true_z.extend(true_x[i:i+5])
        true_z.extend(true_y[i:i+5])
    z = torch.stack(z)
    single_z = torch.stack(single_z)
    true_z = torch.stack(true_z)
    #utils.plot_more_images(true_z.squeeze().unsqueeze(-1),title=f"multiple_{a}-{b}_z",save=True)
    utils.plot_more_images(single_z.squeeze().unsqueeze(-1),title=f"multiple_{a}-{b}_single_z",save=False)
    #utils.plot_more_images(z.squeeze().unsqueeze(-1),title=f"multiple_{a}-{b}_z_multiple_transform2",save=False)

def create_dataloader(model_type=MODELS[0], batch_size=32, transform=True):
    """Creates the Dataloader based on which model will be used for training.
    Args:
    model_type(str): String for which model will be used for training.
    batch_size(int): Batch size of the Dataloader.
    transform(bool): A boolean value, depending if transformations should be applied or not.
    Returns:
    train_dataloader(Dataloader): The Training Dataloader.
    test_dataloader(Dataloader): The Testing Dataloader.
    classes(int): Amount of classes of our model.
    """
    train_dataset, test_dataset = load_reshape_dataset(model_type=model_type,transform=transform)
    classes=train_dataset.classes
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"[DATA] Dataloader (Batch size: {batch_size}, Classes: {classes}) created.")
    return train_dataloader,test_dataloader, classes

def load_reshape_dataset(model_type=MODELS[0],transform=True):
    """Loads and reshapes the dataset based on which model will be used for training.
    Args:
    model_type(str): String for which model will be used for training.
    transform(bool): A boolean value, depending if transformations should be applied or not.
    Returns:
    train_dataset(Dataset): The Training Dataset.
    test_dataset(Dataset): The Testing Dataset.
    """

    train_GDM,train_GSL = utils.load_dataset(dataset_name=DATASETS[0], augmented=True)
    test_GDM,test_GSL = utils.load_dataset(dataset_name=DATASETS[1], augmented=True)
    classes=1
    common_transform = [transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1), data_transformations.RotationTransform(rotation_angle=90)]
    x_transform= [data_transformations.NoiseTransform(),data_transformations.SshapeTransform(),data_transformations.GridTransform(),data_transformations.RandomTransform()]
    #common_transform = [transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1)]

    if not transform:
        common_transform = None
        x_transform= None
    if(model_type==MODELS[0] or model_type==MODELS[2]):
        classes=None
        train_GSL = train_GSL.reshape(train_GSL.shape[0], -1)
        test_GSL = test_GSL.reshape(test_GSL.shape[0], -1)
        train_dataset = VGGDataset(train_GDM, train_GSL, classes, common_transform=common_transform, x_transform=x_transform)
        test_dataset = VGGDataset(test_GDM, test_GSL, classes, common_transform=common_transform, x_transform=x_transform)
        return train_dataset, test_dataset
        #print(f"[RESHAPE] Model_Type: {model_type}, train_GSL: {train_GSL.shape}, test_GSL: {test_GSL.shape}")
    #print(f"[RESHAPE] train_GDM: {train_GDM.shape}, test_GDM: {test_GDM.shape}")
    
    train_dataset = EncDecDataset(train_GDM, train_GSL, classes, common_transform=common_transform, x_transform=x_transform)
    test_dataset = EncDecDataset(test_GDM, test_GSL, classes, common_transform=common_transform, x_transform=x_transform)

    return train_dataset, test_dataset






class EncDecDataset(torch.utils.data.Dataset):
    """Dataset with X,y,classes for the Encoder Decoder model"""
    def __init__(self, X, y, classes=None, common_transform=None, x_transform=None):
        self.X = X
        self.y = y
        if classes is None:
            self.classes = y.shape[-1]
        else:
            self.classes = classes
        self.common_transform = common_transform
        self.x_transform = x_transform

    def __getitem__(self, index):
        x, y = self.X[index], self.y[index]
        #mask = self.mask[index]
        if self.common_transform:
            randomize=torch.rand(len(self.common_transform))            
            #print(randomize)
            #print(randomize[0])
            if randomize[0] < 0.5:
                x, y = self.common_transform[0](x), self.common_transform[0](y)
                #print("horizontal")
              #  print(f"[INFO] Applied transform: {self.common_transform[0]}")
            if randomize[1] < 0.5:
                x, y = self.common_transform[1](x), self.common_transform[1](y)
                #print("vertical")
               # print(f"[INFO] Applied transform: {self.common_transform[1]}")
            if randomize[2]>0.25:
                x, y = self.common_transform[2](x,y)
                #print(90)
                if randomize[2]>0.5:
                    x, y = self.common_transform[2](x,y)           
                    #print(180)
                    if randomize[2]>0.75:
                        x, y = self.common_transform[2](x,y)
                        #print(270)
        #print("---------------------------------")
        if self.x_transform:
            randomize=torch.rand(2)#torch.rand(len(self.x_transform))  
            if randomize[0] < 0.5:
                x = self.x_transform[0](x)     # s-shape 1-0.3=0.7=70% 
            rand_value= int(10*max(randomize[0],0.1))
            if randomize[1] < 0.8:
                x = self.x_transform[1](x,distance=rand_value) # grid 0.3-0.2=0.1=10%    -> 20% nothing 
            elif randomize[1] < 0.85:
                x = self.x_transform[2](x,distance=rand_value) 
            elif randomize[1] < 0.90:
                x = self.x_transform[3](x,distance=rand_value)                       
        #print("----------")                    
        return x, y
    
    def __len__(self):
        return len(self.y)


class VGGDataset(torch.utils.data.Dataset):
    """Dataset with X,y,classes for the VGG model"""
    def __init__(self, X, y, classes=None, common_transform=None, x_transform=None):
        self.X = X
        self.y = y
        if classes is None:
            self.classes = y.shape[-1]
        else:
            self.classes = classes
        self.common_transform = common_transform
        self.x_transform = x_transform

    def __getitem__(self, index):
        x, y = self.X[index], self.y[index]
        y=y.reshape(1,64,64)
        if self.common_transform:
            randomize=torch.rand(len(self.common_transform))            
            #print(randomize)
            #print(randomize[0])
            if randomize[0] < 0.5:
                x, y = self.common_transform[0](x), self.common_transform[0](y)
                #print("horizontal")
              #  print(f"[INFO] Applied transform: {self.common_transform[0]}")
            if randomize[1] < 0.5:
                x, y = self.common_transform[1](x), self.common_transform[1](y)
                #print("vertical")
               # print(f"[INFO] Applied transform: {self.common_transform[1]}")
            if randomize[2]>0.25:
                x, y = self.common_transform[2](x,y)
                #print(90)
                if randomize[2]>0.5:
                    x, y = self.common_transform[2](x,y)           
                    #print(180)
                    if randomize[2]>0.75:
                        x, y = self.common_transform[2](x,y)
                        #print(270)
        #print("---------------------------------")
        if self.x_transform:
            #noise TO DO
            randomize=torch.rand(2)#torch.rand(len(self.x_transform))  
            if randomize[0] < 0.5:
                x = self.x_transform[0](x)     # s-shape 1-0.3=0.7=70% 
            rand_value= int(10*max(randomize[0],0.1))
            if randomize[1] < 0.8:
                x = self.x_transform[1](x,distance=rand_value) # grid 0.3-0.2=0.1=10%    -> 20% nothing 
            elif randomize[1] < 0.85:
                x = self.x_transform[2](x,distance=rand_value) 
            elif randomize[1] < 0.90:
                x = self.x_transform[3](x,distance=rand_value)                  
                
       # print("---------------------------------")
        #print(f"{y.shape}")
        y=y.reshape(y.shape[0], -1).squeeze()
        #print(f"{y.shape}")
        return x, y
    
    def __len__(self):
        return len(self.y)
    




class SuperDataset(torch.utils.data.Dataset):
    """Dataset with X,y,classes for testing purposes"""
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

