"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import utils
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import numpy as np
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class=torch.sigmoid(y_pred)
        y_pred_class = torch.argmax(y_pred_class.squeeze(1).view(y_pred_class.size(0), -1), dim=1)
        y_index = torch.argmax(y.squeeze(1).view(y.size(0), -1), dim=1)

        train_acc += (y_pred_class == y_index).sum().item()/ len(y_pred)#y.size(0)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    test_approx_acc_1=0
    test_approx_acc_2=0
    test_approx_acc_3=0
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels=torch.sigmoid(test_pred_logits)
            test_pred_labels = torch.argmax(test_pred_labels.squeeze(1).view(test_pred_labels.size(0), -1), dim=1)
            y_index = torch.argmax(y.squeeze(1).view(y.size(0), -1), dim=1)
            
            #test_pred_labels = torch.argmax(torch.sigmoid(test_pred_logits), dim=0)#test_pred_logits.argmax(dim=1)
            #y_index=torch.argmax(y,dim=0)
            #test_pred_labels = torch.argmax(torch.softmax(test_pred_logits,dim=1), dim=1)#test_pred_logits.argmax(dim=1)
            #y_index=torch.argmax(y,dim=1)
           # print(f"[INFO] test_pred_logits: {test_pred_logits}")
            #print(f"[INFO] y: {y}")            
            #print(f"[INFO] test_pred_labels: {test_pred_labels}")
            #print(f"[INFO] y_index: {y_index}")
            test_acc += ((test_pred_labels == y_index).sum().item()/len(test_pred_logits))
            test_approx_acc_1+= approximate_accuracy(y_true=y_index,y_predicted=test_pred_labels,height=X.shape[-2],distance=1)
            test_approx_acc_2+= approximate_accuracy(y_true=y_index,y_predicted=test_pred_labels,height=X.shape[-2],distance=2)
            test_approx_acc_3+= approximate_accuracy(y_true=y_index,y_predicted=test_pred_labels,height=X.shape[-2],distance=3)
            

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    test_approx_acc_1 = test_approx_acc_1 / len(dataloader)
    test_approx_acc_2 = test_approx_acc_2 / len(dataloader)
    test_approx_acc_3 = test_approx_acc_3 / len(dataloader)
    return test_loss, test_acc, test_approx_acc_1,test_approx_acc_2,test_approx_acc_3

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          transform: True) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """


    
    # Make sure model on target device
    elapsed_old=0
    model.to(device)
    model_name = type(model).__name__
    model,start=utils.load_model(model=model,model_type=model_name,device=device,transform=transform)
    start2=utils.load_random(model_type=model_name,device=device)
    if(start!=start2):
        print("[ERROR] Start not the same!")
        return
    results=utils.load_loss(model_type=model_name,device=device)
    # Loop through training and testing steps for a number of epochs
    with tqdm(total=start-epochs) as t:
        for epoch in tqdm(range(start, epochs)):
            train_loss, train_acc = train_step(model=model,
                                              dataloader=train_dataloader,
                                              loss_fn=loss_fn,
                                              optimizer=optimizer,
                                              device=device)
            test_loss, test_acc,test_approx_acc_1,test_approx_acc_2,test_approx_acc_3 = test_step(model=model,
              dataloader=test_dataloader,
              loss_fn=loss_fn,
              device=device)


            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            #Logs of the results
            
            logging.info(f"[TRAINING] Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")
            logging.info(f"[ACCURACY] test_acc: {test_acc:.4f} | test_approx_acc_1: {test_approx_acc_1:.4f} | test_approx_acc_2: {test_approx_acc_2:.4f} | test_approx_acc_3: {test_approx_acc_3:.4f}")
            
            elapsed = t.format_dict['elapsed']
            elapsed_str = t.format_interval(elapsed)
            epoch_duration=elapsed-elapsed_old
            elapsed_old=elapsed
            epoch_duration_str = t.format_interval(epoch_duration)
            logging.info(f"[TRAINING] Elapsed: {elapsed_str} | Epoch Duration: {epoch_duration_str}")
            
            utils.save_model(model=model,model_type=model_name,epoch=epoch+1,device=device,transform=transform)        
            utils.save_random(model_name,epoch+1,device)
            utils.save_loss(results,model_name,device)


    # Return the filled results at the end of the epochs
    return results



def approximate_accuracy(y_true, y_predicted, height, distance):
    y_true_height = torch.div(y_true, height, rounding_mode='floor')
    y_true_width = y_true % height
    y_predicted_height = torch.div(y_predicted, height, rounding_mode='floor')
    y_predicted_width = y_predicted % height

    # Calculate the valid neighborhood bounds for each predicted point
    height_min = torch.clamp(y_predicted_height - distance, min=0)
    height_max = torch.clamp(y_predicted_height + distance, max=height - 1)
    width_min = torch.clamp(y_predicted_width - distance, min=0)
    width_max = torch.clamp(y_predicted_width + distance, max=height - 1)

    # Check if true points fall within the corresponding neighborhood
    matches = (
        (y_true_height >= height_min) & (y_true_height <= height_max) &
        (y_true_width >= width_min) & (y_true_width <= width_max)
    )

    # Calculate accuracy
    accuracy = (matches.sum().item() / len(y_true))
    return accuracy
