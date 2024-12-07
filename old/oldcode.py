
def transform_datasets_with_distinctive_source(dataset_type):
    for dataset in DATASETS:
        dataset_GDM,dataset_GSL=utils.load_data(dataset)
        coordinates=datatransformer.find_distinctive_source(dataset_GSL)
        coordinates=torch.IntTensor(coordinates)
        x,y=coordinates[:,0],coordinates[:,1]
        dataset_GSL=dataset_GSL[:,:,x,y]
        dataset_GSL = dataset_GSL.reshape(-1, dataset_GSL.shape[-1])
        dataset_GDM = dataset_GDM.reshape(-1, 1, dataset_GDM.shape[-2],dataset_GDM.shape[-1])
        print(f"[INFO] Dataset GSL shape: {dataset_GSL.shape}")
        print(f"[INFO] Dataset GDM shape: {dataset_GDM.shape}")
        print(f"[INFO] Dataset saved at: data/MyTensor/datasets_{dataset_type}/{dataset}.pt")
        torch.save({'X': dataset_GDM, 'y':dataset_GSL},f"data/MyTensor/datasets_{dataset_type}/{dataset}.pt")


def transform_datasets_flattened(dataset_type):
    for dataset in DATASETS:
        dataset_GDM,dataset_GSL=utils.load_data(dataset)
        dataset_GSL = dataset_GSL.reshape(-1, dataset_GSL.shape[-1]*dataset_GSL.shape[-2])
        dataset_GDM = dataset_GDM.reshape(-1, 1, dataset_GDM.shape[-2],dataset_GDM.shape[-1])
        print(f"[INFO] Dataset GSL shape: {dataset_GSL.shape}")
        print(f"[INFO] Dataset GDM shape: {dataset_GDM.shape}")
        print(f"[INFO] Dataset saved at: data/MyTensor/datasets_{dataset_type}/{dataset}.pt")
        torch.save({'X': dataset_GDM, 'y':dataset_GSL},f"data/MyTensor/datasets_{dataset_type}/{dataset}.pt")

def transform_datasets_grid(dataset_type):
    for dataset in DATASETS:
        dataset_GDM,dataset_GSL=utils.load_data(dataset)
        dataset_GSL=datatransformer.create_grid(dataset_GSL)
        #transformed_dataset = np.zeros_like(dataset_GSL)
        #x,y=coordinates[:,0],coordinates[:,1]
        #transformed_dataset[:,:,::x,::y]=dataset_GSL[:,:,::x,::y]
        #dataset_GSL=transformed_dataset
        dataset_GSL = dataset_GSL.reshape(-1, dataset_GSL.shape[-1]*dataset_GSL.shape[-2])
        dataset_GDM = dataset_GDM.reshape(-1, 1, dataset_GDM.shape[-2],dataset_GDM.shape[-1])
        print(f"[INFO] Dataset GSL shape: {dataset_GSL.shape}")
        print(f"[INFO] Dataset GDM shape: {dataset_GDM.shape}")
        print(f"[INFO] Dataset saved at: data/MyTensor/datasets_{dataset_type}/{dataset}.pt")
        torch.save({'X': dataset_GDM, 'y':dataset_GSL},f"data/MyTensor/datasets_{dataset_type}/{dataset}.pt")













def create_coordinates_s_shape2(dataset_GDM,distance=5,pad=2,start_left=False):
    dimensions=[dataset_GDM[-2],dataset_GDM[-1]]
    coordinates = []
    x=pad
    y=pad
    while(y<dimensions[1]-pad):
        #left to right
        if(start_left):
            while(x<=dimensions[0]-pad):
                coordinates.append([x,y])
                x+=1
            y_max=y+distance-1
            x-=1
            if(y_max>dimensions[1]-pad):
                y_max=dimensions[1]-pad
            while(y<y_max):
                y+=1
                coordinates.append([x,y])
            start_left=False
            y+=1
        else:
        # right to left
            x=dimensions[0]-pad
            while(x>=pad):                
                coordinates.append([x,y])
                x-=1
            y_max=y+distance-1
            if(y_max>dimensions[1]-pad):
                y_max=dimensions[1]-pad
            x+=1
            while(y<y_max):
                y+=1
                coordinates.append([x,y])
            start_left=True
            y+=1
    return coordinates



def create_coordinates_s_shape(dataset_GDM,distance=5,pad=2,start_left=True):
    dimensions=[dataset_GDM[-2],dataset_GDM[-1]]
    coordinates=[]
    x=pad
    count=1
    while(x <= dimensions[0]-pad):
        y=pad
        while(y <= dimensions[1]-pad):
            coordinates.append([x,y])
            y += distance
        x += 1
    if(start_left):
        x=pad
        while(x <= dimensions[0]-pad):
            y=pad+1
            while(y <= dimensions[1]-pad):
                while(y<=pad+distance*count):
                    coordinates.append([x,y])
                    y+=1
                count+=2
            x+=dimensions[0]-2*pad
    return coordinates
