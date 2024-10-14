import unittest
import numpy as np
import create_dataset
import random
import torch
import preprocess_data
import create_model
import torch.nn as nn
import torch.optim as optim





def main():
    save_model_decoder()
    #save_model()
    #model=load_model()
    #test_model(model)


def save_model_simple():
    model = create_model.DeepNN()
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    x=preprocess_data.load_preprocessed_data()
    #x = create_dataset.load_combined_dataset("train",True)
    y=create_dataset.load_combined_dataset("train",False)    
    x=create_batches(x,10)
    y=create_batches(y,10)
    for batch in range(x.shape[0]):
        y_out = y[batch].view(10, -1)  # Shape: (10, 1500)
        optimizer.zero_grad()  # Zero gradients
        output = model(x[batch])  # Forward pass
        loss = criterion(output, y_out)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()
        print(f'Loss: {loss.item()} , Batch: {batch}')
    torch.save(model, "data/model/model.pt")
    print(f"saved  model")

def save_model_decoder():
    model = create_model.DecoderNet()
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    x=preprocess_data.load_preprocessed_data()
    #x = create_dataset.load_combined_dataset("train",True)
    y=create_dataset.load_combined_dataset("train",False)    
    x=create_batches(x,10)
    y=create_batches(y,10)
    for batch in range(x.shape[0]):
        y_out = y[batch].view(10, -1)  # Shape: (10, 1500)
        optimizer.zero_grad()  # Zero gradients
        output = model(x[batch])  # Forward pass
        loss = criterion(output, y_out)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()
        print(f'Loss: {loss.item()} , Batch: {batch}')
    torch.save(model, "data/model/model.pt")
    print(f"saved  model")





def load_model():
    return torch.load("data/model/model.pt")
    

def test_model(model):
    SEQUENCE=[2,100]# specify which image to be shown in data of x,y->[x,y,30,25]   example:39,2
    SIZE=[6,5]#size of plots
    x=preprocess_data.random_cell_generator("test",True)    
    create_dataset.show_as_image_sequence(x, SEQUENCE,SIZE)
    x=x[0,0]
    y_predicted=model(x)
    print(y_predicted.shape)
    print(y_predicted)



def create_batches(dataset,batches):
    new_dim = dataset.shape[1] // batches
    #dataset=dataset.unsqueeze(-1)
    # Reshape the tensor to [420, new_dim, split_size, x, y, z]
    reshaped_tensor = dataset.view(dataset.shape[0], new_dim, batches, dataset.shape[2], dataset.shape[3], dataset.shape[4])
    
    # Combine the first two dimensions [420, new_dim] into a single dimension
    final_tensor = reshaped_tensor.permute(1, 0, 2, 3, 4, 5).reshape(-1, batches, dataset.shape[2], dataset.shape[3], dataset.shape[4])
    final_tensor = final_tensor.type(torch.float)
    return final_tensor








if __name__ == "__main__":
    main()



