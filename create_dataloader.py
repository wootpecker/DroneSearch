import create_dataset
from torch.utils.data import Dataset, DataLoader



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

