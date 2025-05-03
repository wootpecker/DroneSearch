from torch.utils.data import Dataset, DataLoader, random_split
import torch
import utils
import logging
from timeit import default_timer as timer 
from torchvision import transforms
import data_transformations
import create_dataset
import matplotlib.pyplot as plt
from logs import logger

MODELS = ["VGG8", "UnetS", "VGGVariation"]
DATASETS=["train","test"]
COMMON_TRANSFORM=[transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1), data_transformations.RotationTransform(rotation_angle=90),data_transformations.RotationTransform(rotation_angle=180),data_transformations.RotationTransform(rotation_angle=270)]
COMMON_TRANSFORM_PROB=[0.5, 0.5, 0.25, 0.5, 0.75]#HorizontalFlip(50%),VerticalFlip(50%),Rotation90(25%),Rotation180(25%),Rotation270(25%)
X_TRANSFORM=[data_transformations.NoiseTransform(),data_transformations.SshapeTransform(),data_transformations.CageTransform(),data_transformations.GridTransform()]
X_TRANSFORM_PROB=[0.5, 0.4, 0.6, 0.7]#Noise(50%),Sshape(40%),Cage(20%),Grid(10%)
#COMMON_TRANSFORM_PROB=[1,0,0,0,0]
#X_TRANSFORM_PROB=[0, 0, 0, 0]




def main():
    #create_dataloader_distinctive(batch_size=16)
    #test()
    plot_for_BA(model_type=MODELS[1])

 #TESTING:

def plot_for_BA(model_type=MODELS[1]):
    logger.logging_config(logs_save=False)
    global COMMON_TRANSFORM_PROB, X_TRANSFORM_PROB
    COMMON_TRANSFORM_PROB = [0, 0, 0, 0, 0]
    X_TRANSFORM_PROB = [0, 0, 0, 0]
    utils.seed_generator()
    train_GDM,train_GSL,test_GDM,test_GSL=create_dataset.create_dataset(amount_samples=1,window_size=[64,64],save=False)
    train_dataset, test_dataset = load_reshape_dataset(model_type=model_type,transform=True, load=False, train_GDM=train_GDM, train_GSL=train_GSL, test_GDM=test_GDM, test_GSL=test_GSL)
    #train_dataset, test_dataset= load_reshape_dataset(model_type=model_type, transform=True, load=False)
    sample_number=4958
    size=[1,3]
    fig_width = 3 * 2  # 4 columns × 2 inches per image
    fig_height =  2  # 5 rows × 2 inches per image
    f, arr = plt.subplots(size[0],size[1], figsize=(fig_width, fig_height)) 
    for j in range(arr.shape[0]):
        #print(i)
        
        COMMON_TRANSFORM_PROB=[0,0,0,0,0]
        if j!=0:        
            COMMON_TRANSFORM_PROB[j]=1
        X_TRANSFORM_PROB=[0, 0, 0, 0]
        sample=train_dataset.__getitem__(sample_number)
        arr[j].imshow(sample[0].squeeze(0).unsqueeze(-1).numpy(), origin="lower")        



    f.subplots_adjust(hspace =0.4)
   # f.tight_layout()
    plt.subplots_adjust(hspace =0.4)
    plt.show()










    a=0 
    b=5
    if model_type==MODELS[0]:
        max_value = torch.max(temp[0][0])
        print(f"Max value in the dataset: {max_value}")
        if model_type==MODELS[0]:
            temporary=temp[1].reshape(1,temp[0].shape[-2],temp[0].shape[-1])
            single_y.append(temporary)
            #temp[1]=temp[1].reshape(1,temp[0].shape[-2],temp[0].shape[-1])
        else:
            single_y.append(temp[1])
        single_x.append(temp[0])
    single_x = torch.stack(single_x)
    single_y = torch.stack(single_y)
    #print(f"Max value in the dataset: {max_value}")
    #z=train_dataset[a:b]
    single_z = []
    print("new dataset test:")
    train_dataset, test_dataset= load_reshape_dataset(model_type=model_type,transform=False, load=False)
    if model_type==MODELS[0]:
        temp_list=[]
        x=[]
        y=[]
        z=[]
        true_z=[]
        for i in range(a,b):
            temp=train_dataset.__getitem__(i)
            temporary=temp[1].reshape(1,temp[0].shape[-2],temp[0].shape[-1])
            temp_list.append(temporary)
            x.append(temp[0])
            y.append(temporary)
        true_x=x
        true_y=y
        end=b
    else:
        z=train_dataset[a:b]
        x=z[0]
        y=z[1]
        z = []
        true_z=train_dataset[a:b]
        true_x=true_z[0]
        true_y=true_z[1]
        true_z=[]
        end=x.shape[0]
    for i in range(0, end, 5):
        z.extend(x[i:i+5])
        z.extend(y[i:i+5])
        single_z.extend(single_x[i:i+5])
        single_z.extend(single_y[i:i+5])
        true_z.extend(true_x[i:i+5])
        true_z.extend(true_y[i:i+5])

    z = torch.stack(z)
    single_z = torch.stack(single_z)
    true_z = torch.stack(true_z)
    utils.plot_more_images(true_z.squeeze().unsqueeze(-1),title=f"multiple_{a}-{b}_z",save=True)
    utils.plot_more_images(single_z.squeeze().unsqueeze(-1),title=f"multiple_{a}-{b}_single_z",save=True)
    #utils.plot_more_images(z.squeeze().unsqueeze(-1),title=f"multiple_{a}-{b}_z_multiple_transform2",save=False)



# FUNCTIONS:
#amount_samples=8,window_size=[64, 64],train_ratio=0.8, save=True
def create_dataloader(model_type=MODELS[0], batch_size=32, transform=True, mask=False, amount_samples=0, window_size=[64,64]):
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
    #train_GDM=None, train_GSL=None, test_GDM=None, test_GSL=None
    if amount_samples==0:
        train_dataset, test_dataset = load_reshape_dataset(model_type=model_type,transform=transform, load=True, train_GDM=None, train_GSL=None, test_GDM=None, test_GSL=None)
    else:
        train_GDM,train_GSL,test_GDM,test_GSL=create_dataset.create_dataset(amount_samples=amount_samples,window_size=window_size,save=False)
        train_dataset, test_dataset = load_reshape_dataset(model_type=model_type,transform=transform, load=False, train_GDM=train_GDM, train_GSL=train_GSL, test_GDM=test_GDM, test_GSL=test_GSL)
    classes=train_dataset.classes
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logging.info(f"[DATALOADER] Dataloader (Batch size: {batch_size}, Classes: {classes}) created.")

    return train_dataloader,test_dataloader, classes



def load_reshape_dataset(model_type=MODELS[0],transform=True, load=True, train_GDM=None, train_GSL=None, test_GDM=None, test_GSL=None):
    """Loads and reshapes the dataset based on which model will be used for training.
    Args:
    model_type(str): String for which model will be used for training.
    transform(bool): A boolean value, depending if transformations should be applied or not.
    Returns:
    train_dataset(Dataset): The Training Dataset.
    test_dataset(Dataset): The Testing Dataset.
    """
    if load:
        train_GDM,train_GSL = utils.load_dataset(dataset_name=DATASETS[0], augmented=True)
        test_GDM,test_GSL = utils.load_dataset(dataset_name=DATASETS[1], augmented=True)
    classes=1
    common_transform = COMMON_TRANSFORM
    x_transform= X_TRANSFORM#data_transformations.RandomTransform()]
    #common_transform = [transforms.RandomHorizontalFlip(p=1), transforms.RandomVerticalFlip(p=1)]
    #x_transform= None
    #common_transform = [data_transformations.RotationTransform(rotation_angle=90)]
    if not transform:
        common_transform = None
        x_transform= None
      
    if(model_type==MODELS[0] or model_type==MODELS[2]):
        classes=None
        train_GSL = train_GSL.reshape(train_GSL.shape[0], -1)
        test_GSL = test_GSL.reshape(test_GSL.shape[0], -1)
        train_dataset = VGGDataset(train_GDM, train_GSL, classes, common_transform=common_transform, x_transform=x_transform)
        test_dataset = VGGDataset(test_GDM, test_GSL, classes, common_transform=common_transform, x_transform=x_transform)
        common_transform_str=common_transform_to_str(common_transform)
        x_transform_str=x_transform_to_str(x_transform)
        logging.info(f"[DATALOADER] Dataloader: {type(train_dataset).__name__}")
        logging.info(f"[TRANSFORM] XY-Transform: {common_transform_str}") 
        logging.info(f"[TRANSFORM] X-Transform: {(x_transform_str)}")   
        return train_dataset, test_dataset
        #print(f"[RESHAPE] Model_Type: {model_type}, train_GSL: {train_GSL.shape}, test_GSL: {test_GSL.shape}")
    #print(f"[RESHAPE] train_GDM: {train_GDM.shape}, test_GDM: {test_GDM.shape}")
    
    train_dataset = EncDecDataset(train_GDM, train_GSL, classes, common_transform=common_transform, x_transform=x_transform)
    test_dataset = EncDecDataset(test_GDM, test_GSL, classes, common_transform=common_transform, x_transform=x_transform)
    common_transform_str=common_transform_to_str(common_transform)
    x_transform_str=x_transform_to_str(x_transform)
    logging.info(f"[DATALOADER] Dataloader: {type(train_dataset).__name__}")
    logging.info(f"[TRANSFORM] XY-Transform: {common_transform_str}") 
    logging.info(f"[TRANSFORM] X-Transform: {(x_transform_str)}")    
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
            #HorizontalFlip Transform           
            if randomize[0] < COMMON_TRANSFORM_PROB[0]:
                x, y = self.common_transform[0](x), self.common_transform[0](y)
            #VerticalFlip Transform                
            if randomize[1] < COMMON_TRANSFORM_PROB[1]:
                x, y = self.common_transform[1](x), self.common_transform[1](y)
            #Rotation90 Transform
            if randomize[2] < COMMON_TRANSFORM_PROB[2]:
                x, y = self.common_transform[2](x, y)
            #Rotation180 Transform
            elif randomize[2] < COMMON_TRANSFORM_PROB[3]:
                x, y = self.common_transform[3](x, y)
            #Rotation270 Transform
            elif randomize[2] < COMMON_TRANSFORM_PROB[4]:
                x, y = self.common_transform[4](x, y)
        if self.x_transform:
            randomize=torch.rand(2)
            #Noise Transform
            if randomize[0] < X_TRANSFORM_PROB[0]:
                x = self.x_transform[0](x)
            #distance -> distance between measurements
            rand_value= int(max(10*randomize[0],1))
            #Sshape Transform
            if randomize[1] < X_TRANSFORM_PROB[1]:
                x = self.x_transform[1](x,distance=rand_value)   
            #Grid Transform
            elif randomize[1] < X_TRANSFORM_PROB[2]:
                x = self.x_transform[2](x,distance=rand_value) 
            #Cage Transform
            elif randomize[1] < X_TRANSFORM_PROB[3]:
                x = self.x_transform[3](x,distance=rand_value)                                       
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
        #Reshape y, to perform transformations
        y=y.reshape(1,x.shape[-2],x.shape[-1])
        if self.common_transform:
            randomize=torch.rand(len(self.common_transform)) 
           #HorizontalFlip Transform           
            if randomize[0] < COMMON_TRANSFORM_PROB[0]:
                x, y = self.common_transform[0](x), self.common_transform[0](y)
            #VerticalFlip Transform                
            if randomize[1] < COMMON_TRANSFORM_PROB[1]:
                x, y = self.common_transform[1](x), self.common_transform[1](y)
            #Rotation90 Transform
            if randomize[2] < COMMON_TRANSFORM_PROB[2]:
                x, y = self.common_transform[2](x, y)
            #Rotation180 Transform
            elif randomize[2] < COMMON_TRANSFORM_PROB[3]:
                x, y = self.common_transform[3](x, y)
            #Rotation270 Transform
            elif randomize[2] < COMMON_TRANSFORM_PROB[4]:
                x, y = self.common_transform[4](x, y)
        if self.x_transform:
            randomize=torch.rand(2)
            #Noise Transform
            if randomize[0] < X_TRANSFORM_PROB[0]:
                x = self.x_transform[0](x)
            #distance -> distance between measurements
            rand_value= int(max(10*randomize[0],1))
            #Sshape Transform
            if randomize[1] < X_TRANSFORM_PROB[1]:
                x = self.x_transform[1](x,distance=rand_value)   
            #Grid Transform
            elif randomize[1] < X_TRANSFORM_PROB[2]:
                x = self.x_transform[2](x,distance=rand_value) 
            #Cage Transform
            elif randomize[1] < X_TRANSFORM_PROB[3]:
                x = self.x_transform[3](x,distance=rand_value)     
        #Reshape back into original shape                
        y=y.reshape(y.shape[0], -1).squeeze()
        return x, y
    
    def __len__(self):
        return len(self.y)
    
def common_transform_to_str(common_transform):
    common_transform_str=[]
    if common_transform is None:
        return None    
    if any(isinstance(transform, transforms.RandomHorizontalFlip) for transform in common_transform):
        common_transform_str.append(f"HorizontalFlip: {int(COMMON_TRANSFORM_PROB[0]*100)}%,")
    if any(isinstance(transform, transforms.RandomVerticalFlip) for transform in common_transform):
        common_transform_str.append(f"VerticalFlip: {int(COMMON_TRANSFORM_PROB[1]*100)}%,")        
    if any(isinstance(transform, data_transformations.RotationTransform) and transform.rotation_angle==90 for transform in common_transform):
        common_transform_str.append(f"Rotation90: {int(COMMON_TRANSFORM_PROB[2]*100)}%,")        
    if any(isinstance(transform, data_transformations.RotationTransform) and transform.rotation_angle==180 for transform in common_transform):
        common_transform_str.append(f"Rotation180: {int(round((COMMON_TRANSFORM_PROB[3]-COMMON_TRANSFORM_PROB[2])*100))}%,")
    if any(isinstance(transform, data_transformations.RotationTransform) and transform.rotation_angle==90 for transform in common_transform):
        common_transform_str.append(f"Rotation270: {int(round((COMMON_TRANSFORM_PROB[4]-COMMON_TRANSFORM_PROB[3])*100))}%")
    return common_transform_str

        
def x_transform_to_str(x_transform):
    if x_transform is None:
        return None
    x_transform_str=[]
    if any(isinstance(transform, data_transformations.NoiseTransform) for transform in x_transform):
        x_transform_str.append(f"Noise: {int(X_TRANSFORM_PROB[0]*100)}%,")
    if any(isinstance(transform, data_transformations.SshapeTransform) for transform in x_transform):
        x_transform_str.append(f"SShape: {int(X_TRANSFORM_PROB[1]*100)}%,")
    if any(isinstance(transform, data_transformations.CageTransform) for transform in x_transform):
        x_transform_str.append(f"Cage: {int(round((X_TRANSFORM_PROB[2]-X_TRANSFORM_PROB[1])*100))}%,")
    if any(isinstance(transform, data_transformations.GridTransform) for transform in x_transform):
        x_transform_str.append(f"Grid: {int(round((X_TRANSFORM_PROB[3]-X_TRANSFORM_PROB[2])*100))}%,")       
    return x_transform_str    



if __name__ == "__main__":
    main()

