import create_dataset
from torch.utils.data import Dataset, DataLoader

DATA="train" #train,valid,test                   Old/30x25/
TRANSFORMED=False #reduce 30x25->6x5
SEQUENCE=[2,100]# specify which image to be shown in data of x,y->[x,y,30,25]   example:39,2
SIZE=[6,5]#size of plots


def main():
    #create_dataloader_distinctive(batch_size=16)
    create_dataloader(batch_size=32)




def create_dataloader(dataloader_type="Flattened", batch_size=32):
    train_dataset = create_dataset.SuperDataset(f"data/MyTensor/datasets_{dataloader_type}/train.pt")
    test_dataset = create_dataset.SuperDataset(f"data/MyTensor/datasets_{dataloader_type}/test.pt")
    valid_dataset = create_dataset.SuperDataset(f"data/MyTensor/datasets_{dataloader_type}/valid.pt")
    classes=train_dataset.classes
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    print(f"[INFO] Dataloader (Type: {dataloader_type}, Batch size: {batch_size}) created.")
    return train_dataloader,test_dataloader,valid_dataloader, classes

if __name__ == "__main__":
    main()

