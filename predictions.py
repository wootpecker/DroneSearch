"""
Utility functions to make predictions.

Main reference for code creation: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set 
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List, Tuple
import create_dataloader,model_builder,utils
from tqdm.auto import tqdm
import torchmetrics
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import random
import pandas as pd

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=32
LOAD_SEED=16923
EASY_ACCESS=[1,1]


def main():
    utils.seed_generator(SEED=LOAD_SEED)
    dataloader_types=["Flattened","Distinctive"] #flattened x:30x25 -> y:750, distinctive x:30x25 -> y:30 
    model_types=["VGG24","CNN","VGGVariation"] 

    do_predictions(dataloader_type=dataloader_types[EASY_ACCESS[0]],model_type=model_types[EASY_ACCESS[1]])

    #do_predictions_confusion_matrix(flattened=True)
    #do_predictions_confusion_matrix(model="CNNwithDistinctiveVGG")
    #plot_confusionmatrix




def do_predictions(dataloader_type="Flattened",model_type= "VGG24"):
    """Makes prediction with a model and plots 5 different test samples with respective results.
    

    Args:
    model(string): Type of model to be used for predicting.
    dataloader(string): Type of dataset to be used for predicting.

    Returns:
    Plot of 5 samples
    Plot of confusion matrix
    """
    utils.seed_generator(SEED=LOAD_SEED)
    train_dataloader,test_dataloader,valid_dataloader,classes = create_dataloader.create_dataloader(dataloader_type=dataloader_type, batch_size=BATCH_SIZE)
    model = model_builder.choose_model(model_type=model_type,output_shape=classes,device=device)
    model=utils.load_model(model= model, target_dir=dataloader_type, model_type=model_type, device=device)

    y_pred,y_list,X_list,y_logit_list,y_preds_percent=make_prediction_all_results(model=model,test_dataloader=test_dataloader)
    print_metrics(y_pred,y_list,y_preds_percent,classes)
    make_plots(y_pred,X_list,y_list,y_logit_list,y_preds_percent)  











def make_predictions(model,test_dataloader):
    y_preds = []
    y_list=[]
    model.eval()
    with torch.inference_mode():
      for batch, (X, y) in tqdm(enumerate(test_dataloader), desc="Making predictions"):
        # Send data and targets to target device
        y_list.append(y.argmax(dim=1).cpu())
        X, y = X.to(device), y.to(device)
        # Do the forward pass
        y_logit = model(X)
        # Turn predictions from logits -> prediction probabilities -> predictions labels
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) 
        y_preds.append(y_pred.cpu())
    y_list_tensor=torch.cat(y_list)
    y_pred_tensor = torch.cat(y_preds)
    return y_pred_tensor,y_list_tensor





def make_prediction_all_results(model,test_dataloader):
    y_pred_list = []
    y_list=[]
    X_list=[]
    y_logit_list=[]
    y_preds_percent=[]
    model.eval()
    with torch.inference_mode():
      for batch, (X, y) in tqdm(enumerate(test_dataloader), desc="Making predictions"):
        # Send data and targets to target device
        y_list.append(y.cpu())
        X_list.append(X.cpu())
        X, y = X.to(device), y.to(device)
        # Do the forward pass
        y_logit = model(X)
        # Turn predictions from logits -> prediction probabilities -> predictions labels
        y_pred_percentage=torch.softmax(y_logit, dim=1)
        y_pred = torch.argmax(y_pred_percentage, dim=1) 
        y_pred_list.append(y_pred.cpu())
        y_logit_list.append(y_logit.cpu())
        y_preds_percent.append(y_pred_percentage.cpu())
    # Concatenate list of predictions into a tensor
    y_list_tensor=torch.cat(y_list)
    y_pred_tensor = torch.cat(y_pred_list)
    X_list_tensor=torch.cat(X_list)
    y_logit_list_tensor=torch.cat(y_logit_list)
    y_preds_percent_tensor=torch.cat(y_preds_percent)
    return y_pred_tensor,y_list_tensor,X_list_tensor,y_logit_list_tensor,y_preds_percent_tensor
    




def make_plots(y_pred,X_list,y_list,y_logit_list,y_preds_percent):
    #def show_as_image_sequence_batch(dataset, predicted_dataset):
    #results=[X_list,y_list,y_logit_list,y_preds_percent]
    y_list,y_logit_list,y_preds_percent=reshape_tensor(y_list,y_logit_list,y_preds_percent)
    size=[5,4]
    f, arr = plt.subplots(size[0],size[1]) 
    random_samples=random.sample(range(X_list.shape[0]),k=5)
    for j in range(arr.shape[0]):
        i=random_samples[j]
        print(i)
        arr[j,0].imshow(X_list[i].squeeze(0).unsqueeze(-1).numpy())
        arr[j,0].set(xlabel=f"Sample: {i}, X")
        arr[j,1].imshow(y_list[i].squeeze(0).unsqueeze(-1).numpy())
        arr[j,1].set(xlabel=f"Sample: {i}, y")
        arr[j,2].imshow(y_logit_list[i].squeeze(0).unsqueeze(-1).numpy())
        arr[j,2].set(xlabel=f"Sample: {i}, y_logit")
        arr[j,3].imshow(y_preds_percent[i].squeeze(0).unsqueeze(-1).numpy())
        arr[j,3].set(xlabel=f"Sample: {i}, y_preds_percent")
        #plt.xticks([])
        #plt.yticks([])
   # f.tight_layout()
    plt.subplots_adjust(hspace =0.4)
    plt.show()

def reshape_tensor(y_list,y_logit_list,y_preds_percent):
    if(y_list.shape[-1]>30):
        y_list=torch.reshape(y_list, (y_list.shape[0], 30, 25))
        y_logit_list=torch.reshape(y_logit_list, (y_logit_list.shape[0], 30, 25))
        y_preds_percent=torch.reshape(y_preds_percent, (y_preds_percent.shape[0], 30, 25))
    else:
        y_list=torch.reshape(y_list, (y_list.shape[0], 6, 5))
        y_logit_list=torch.reshape(y_logit_list, (y_logit_list.shape[0], 6, 5))
        y_preds_percent=torch.reshape(y_preds_percent, (y_preds_percent.shape[0], 6, 5))
    return y_list,y_logit_list,y_preds_percent






def print_metrics(y_pred,y_list,y_preds_percent,classes):
    y_list=torch.argmax(y_list,dim=1)
    acc = torchmetrics.functional.accuracy(y_pred, y_list, task="multiclass", num_classes=classes)
    #print(acc)
    f1= torchmetrics.F1Score(task="multiclass", num_classes=classes)
    acc2= torchmetrics.Accuracy("multiclass",num_classes=classes)
    ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=classes, average=None)
    confmat = ConfusionMatrix(task="multiclass", num_classes=classes)
    confmat_tensor=confmat(y_pred,y_list)

    print(
      f"Accuracy: {acc} | \n"
      f"Accuracy Function: {acc2(y_pred,y_list)} | \n"
      f"F1 Function: {f1(y_pred,y_list)} | \n"
      #f"Average Precision Function: {ap(y_preds_percent,y_list)} | \n"
      #f"Confusion Matrix: {confmat(y_pred,y_list)} \n"
      f"Confusion Matrix shape: {confmat_tensor.shape}"
    )    
    df_cfm = pd.DataFrame(confmat(y_pred,y_list).numpy(), index = range(classes), columns = range(classes))
    
    #df_cfm.to_csv('data/confusion_matrix/cfmtest.csv')
    #print("saved")
    if(classes>100):
        #plt.figure(figsize = (10,7))
        plt.imshow(df_cfm.to_numpy())
    else:
        #plot confusion matrix
        plot_confusionmatrix(confmat_tensor)
        #more stylish plot
        #plot_conf_sklearn(y_pred_tensor=y_pred,y_target_tensor=y_list,classes=classes)

#Plot Confusion_Matrix Helper functions
def plot_confusionmatrix(confmat_tensor):
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
        class_names=range(confmat_tensor.shape[0]), # turn the row and column labels into class names
        #figsize=(10, 7)
    )

def plot_conf_sklearn(y_pred_tensor, y_target_tensor, classes):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_target_tensor, y_pred_tensor, labels=range(classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=range(classes))
    disp.plot()



















if __name__ == "__main__":
    main()