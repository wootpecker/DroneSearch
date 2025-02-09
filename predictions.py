"""
Utility functions to make predictions.

Main reference for code creation: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set 
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
#import ..model_dataloader as model_dataloader
from logs import logger
import model_dataloader
import model_builder
import utils
from tqdm.auto import tqdm
import torchmetrics
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import random
import pandas as pd
import math
import logging
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPES = ["VGG", "EncoderDecoder", "VGGVariation"]

HYPER_PARAMETERS = {
              "SAVE_DATASET": False,
               "TRANSFORM": False,
               "MODEL_TYPES": ["VGG", "EncoderDecoder", "VGGVariation"],
               "LOGS_SAVE": True,
               "AMOUNT_SAMPLES": 16,
               "WINDOW_SIZE": [64,64]
  }



TRAINING_PARAMETERS = {
              "NUM_EPOCHS": 50,
               "BATCH_SIZE": 128,
               "LEARNING_RATE": 0.001,
               "LOAD_SEED": 16923,
               "TRAIN_SEED": 42
  }

MODEL_TO_TEST=(HYPER_PARAMETERS['MODEL_TYPES'][1],HYPER_PARAMETERS['MODEL_TYPES'][2])


def main():
    
    logger.logging_config(logs_save=False)
    #model_to_test=(HYPER_PARAMETERS['MODEL_TYPES'][1],HYPER_PARAMETERS['MODEL_TYPES'][2])
    #model_type=HYPER_PARAMETERS['MODEL_TYPES'][2],
    result_dic=[]
    for model_type in MODEL_TO_TEST:
        accuracy_results=do_predictions(model_type=model_type)
        logging.info(f"[ACCURACY] Results: {accuracy_results}")
        #print(accuracy_results)
        result_dic.append(accuracy_results)
    plot_model_accuracies(result_dic)
    #do_predictions_confusion_matrix(flattened=True)
    #do_predictions_confusion_matrix(model="CNNwithDistinctiveVGG")
    #plot_confusionmatrix


def plot_model_accuracies(result_dic):
    accuracy_amount =  range(1,11)
    # Setup a plot 
    plt.figure(figsize=(15, 7))
    for x in range(len(result_dic)):
        model=result_dic[x]
        accuracy_amount =  range(1,len(model['approximate_accuracy'])+1)
        plt.plot(accuracy_amount, model['approximate_accuracy'], label=f'{MODEL_TO_TEST[x]} Approximate Accuracy', color=f'C{x}')
        plt.plot(accuracy_amount, model['topx_accuracy'], label=f'{MODEL_TO_TEST[x]} Top Values Accuracy', linestyle='--', color=f'C{x}')
    # plt.plot(accuracy_amount, mean_percentage, label='mean_percentage')
    plt.title('Accuracy')
    plt.xlabel('Accuracy Distance')
    plt.legend()
    plt.show()
    plt.figure(figsize=(15, 7))
    for x in range(len(result_dic)):
        model=result_dic[x]
        accuracy_amount =  range(1,len(model['approximate_accuracy'])+1)
        plt.plot(accuracy_amount, model['mean_percentage'], label=f'{MODEL_TO_TEST[x]} Mean Percentage', color=f'C{x}')
        #plt.plot(accuracy_amount, model['topx_accuracy'], label=f'{model["model_type"]} Top Values Accuracy')
    # plt.plot(accuracy_amount, mean_percentage, label='mean_percentage')
    plt.title('Confidence Comparison')
    plt.xlabel('Accuracy Distance')
    plt.legend()
    plt.show()    
    return result_dic

def do_predictions(model_type= "VGG"):
    """Makes prediction with a model and plots 5 different test samples with respective results.
    

    Args:
    model(string): Type of model to be used for predicting.
    dataloader(string): Type of dataset to be used for predicting.

    Returns:
    Plot of 5 samples
    Plot of confusion matrix
    """
    utils.seed_generator(SEED=TRAINING_PARAMETERS['LOAD_SEED'])
    train_dataloader,test_dataloader,classes = model_dataloader.create_dataloader(model_type=model_type, batch_size=TRAINING_PARAMETERS['BATCH_SIZE'], transform=HYPER_PARAMETERS['TRANSFORM'], amount_samples=HYPER_PARAMETERS['AMOUNT_SAMPLES'], window_size=HYPER_PARAMETERS['WINDOW_SIZE'])
    model = model_builder.choose_model(model_type=model_type,output_shape=classes,device=device,window_size=HYPER_PARAMETERS['WINDOW_SIZE'])
    model,_=utils.load_model(model= model, model_type=model_type, device=device)
    utils.seed_generator(SEED=TRAINING_PARAMETERS['LOAD_SEED'])
    y_pred,y_list,X_list,y_logit_list,y_preds_percent=make_prediction_all_results(model_type=model_type,model=model,test_dataloader=test_dataloader)
    accuracy_results=print_metrics(y_pred,y_list,y_preds_percent,classes,model_type)
    y_list,y_logit_list,y_preds_percent=reshape_tensor(y_list=y_list,y_logit_list=y_logit_list,y_preds_percent=y_preds_percent)
    make_plots(y_pred,X_list,y_list,y_logit_list,y_preds_percent)  
    return accuracy_results










def make_prediction_all_results(model,test_dataloader,model_type):
    y_pred_list = []
    y_list=[]
    X_list=[]
    y_logit_list=[]
    y_preds_percent=[]
    model.eval()
    logging.info(f"Making predictions:")

    with torch.inference_mode():
      for batch, (X, y) in tqdm(enumerate(test_dataloader), desc="Working", total=len(test_dataloader)):
        # Send data and targets to target device
        y_list.append(y.cpu())
        X_list.append(X.cpu())
        X, y = X.to(device), y.to(device)
        # Do the forward pass
        y_logit = model(X)
        # Turn predictions from logits -> prediction probabilities -> predictions labels
        if(model_type==MODEL_TYPES[1]):
            y_pred_percentage = torch.sigmoid(y_logit)
        else:
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
    size=[4,4]
    f, arr = plt.subplots(size[0],size[1]) 
    random_samples=random.sample(range(X_list.shape[0]),k=size[0])
    for j in range(arr.shape[0]):
        i=random_samples[j]
        #print(i)
        arr[j,0].imshow(X_list[i].squeeze(0).unsqueeze(-1).numpy())
        arr[j,0].set(xlabel=f"Sample: {i}, X (Input of Model)") 
        max_index = y_list[i].argmax().item()
        max_y, max_x = divmod(max_index,y_list[i].shape[-1])
        arr[j,1].imshow(y_list[i].squeeze(0).unsqueeze(-1).numpy())
        arr[j,1].set(xlabel=f"y (Target), Max Value at ({max_x}, {max_y})")
        arr[j,2].imshow(y_logit_list[i].squeeze(0).unsqueeze(-1).numpy())
        arr[j,2].set(xlabel=f"y_logit (Output of Model)")
        arr[j,3].imshow(y_preds_percent[i].squeeze(0).unsqueeze(-1).numpy())
        #arr[j,3].set(xlabel=f"Sample: {i}, y_preds_percent")
        max_value = y_preds_percent[i].max().item()
        max_index = y_preds_percent[i].argmax().item()
        max_y, max_x = divmod(max_index, y_preds_percent[i].shape[-1])
        arr[j,3].set(xlabel=f"Predicted Percentages, Max Value: {max_value:.2f} at ({max_x}, {max_y})")
        #plt.xticks([])
        #plt.yticks([])
    # f.tight_layout()
    plt.subplots_adjust(hspace =0.4)
    plt.show()

def reshape_tensor(y_list,y_logit_list,y_preds_percent):
    if(len(y_list.shape)>2):
        return y_list,y_logit_list,y_preds_percent
    else:
        y_list=torch.reshape(y_list, (y_list.shape[0], HYPER_PARAMETERS['WINDOW_SIZE'][0], HYPER_PARAMETERS['WINDOW_SIZE'][1]))
        y_logit_list=torch.reshape(y_logit_list, (y_logit_list.shape[0], HYPER_PARAMETERS['WINDOW_SIZE'][0], HYPER_PARAMETERS['WINDOW_SIZE'][1]))
        y_preds_percent=torch.reshape(y_preds_percent, (y_preds_percent.shape[0], HYPER_PARAMETERS['WINDOW_SIZE'][0], HYPER_PARAMETERS['WINDOW_SIZE'][1]))        
    return y_list,y_logit_list,y_preds_percent






def print_metrics(y_pred,y_list,y_preds_percent,classes,model_type):
    if(model_type==MODEL_TYPES[1]):
        y_pred_index = torch.argmax(y_preds_percent.view(y_preds_percent.shape[0], -1), dim=1)  # [batch]
        y_list_index = torch.argmax(y_list.view(y_list.shape[0], -1), dim=1)  # [batch]

        return plot_accuracy_curves(y_list_index,y_pred_index,y_list,y_preds_percent,y_pred.shape[-1])

    y_old=y_list            
    y_list=torch.argmax(y_list,dim=1)
    #acc = torchmetrics.functional.accuracy(y_pred, y_list, task="multiclass", num_classes=classes)
    #print(acc)
    #f1= torchmetrics.F1Score(task="multiclass", num_classes=classes)
    #acc2= torchmetrics.Accuracy("multiclass",num_classes=classes)
    #ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=classes, average=None)


    #print(
      #f"Accuracy: {acc} | \n"
      #f"Accuracy Function: {acc2(y_pred,y_list)} | \n"
      #f"F1 Function: {f1(y_pred,y_list)} | \n"
      #f"Average Precision Function: {ap(y_preds_percent,y_list)} | \n"
      #f"Confusion Matrix: {confmat(y_pred,y_list)} \n"
     # f"Confusion Matrix shape: {confmat_tensor.shape}"
    #)    
    #compare_mistakes(y_pred,y_list)
    return plot_accuracy_curves(y_list,y_pred,y_old,y_preds_percent,math.sqrt(classes))
    #confmat = ConfusionMatrix(task="multiclass", num_classes=classes)
    #confmat_tensor=confmat(y_pred,y_list)

    approx_acc_1=approximate_accuracy(y_list,y_pred,math.sqrt(classes),1)
    approx_acc_2=approximate_accuracy(y_list,y_pred,math.sqrt(classes),2)
    approx_acc_3=approximate_accuracy(y_list,y_pred,math.sqrt(classes),3)
    approx_acc_4=approximate_accuracy(y_list,y_pred,math.sqrt(classes),4)
    approx_acc_5=approximate_accuracy(y_list,y_pred,math.sqrt(classes),5)
    top_acc_2=topx_accuracy(y_old,y_preds_percent,2)
    top_acc_3=topx_accuracy(y_old,y_preds_percent,3)
    top_acc_4=topx_accuracy(y_old,y_preds_percent,4)
    top_acc_5=topx_accuracy(y_old,y_preds_percent,5)
    top_acc_6=topx_accuracy(y_old,y_preds_percent,6)
    print(f"Approximate Accuracy 1: {approx_acc_1} | Approximate Accuracy 2: {approx_acc_2} | Approximate Accuracy 3: {approx_acc_3}  | Approximate Accuracy 4: {approx_acc_4}  | Approximate Accuracy 5: {approx_acc_5} ")
    print(f"Top 2 Accuracy: {top_acc_2} | Top 3 Accuracy: {top_acc_3} | Top 4 Accuracy: {top_acc_4} | Top 5 Accuracy: {top_acc_5} | Top 6 Accuracy: {top_acc_6} ")
    #df_cfm.to_csv('data/confusion_matrix/cfmtest.csv')
    #print("saved")
    if(classes>100):
        #plt.figure(figsize = (10,7))
        df_cfm = pd.DataFrame(confmat(y_pred,y_list).numpy(), index = range(classes), columns = range(classes))
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



def compare_mistakes(y_pred_index,y_list_index):
    mistakes=[]
    correct_values=[]
    for x in range(y_list_index.shape[0]):
        if(y_pred_index[x]!=y_list_index[x]):
            print(
              f"x: {x} | \n"
              f"y_pred_index[x]: {y_pred_index[x]} | \n"
              f"y_list_index[x]: {y_list_index[x]} | \n"
            )
            mistakes.append(x)


def test_topx():
    y_true_list = torch.randint(0, 20, (10, 3, 3))
    y_predicted_list = torch.randint(0, 20, (10, 3, 3))
    mean_percentage,topx_accuracies=topx_accuracy(y_true_list,y_predicted_list,2)
    print("topx_accuracy",topx_accuracies)
    print("mean_percentage",mean_percentage)

def topx_accuracy(y_true_list, y_predicted_list, amount_of_values):
    topx_values, topx_indices = torch.topk(y_predicted_list.view(y_predicted_list.shape[0], -1), amount_of_values, dim=1)
    y_true_max, y_true_indices = torch.max(y_true_list.view(y_true_list.shape[0], -1), dim=1)
    matches= torch.any(y_true_indices[:,None] == topx_indices, dim=1)
    accuracy = matches.float().mean().item()
    mean_percentage=torch.mean(topx_values[:,amount_of_values-1].float())
    return mean_percentage,accuracy





def approximate_accuracy(y_true_list, y_predicted_list, height, distance):
    accuracies = 0
    for y_true, y_predicted in zip(y_true_list, y_predicted_list):
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

        # Calculate accuracy for the current pair
        #accuracy = (matches.sum().item() / len(y_true))
        accuracies+=matches
    
    
    return accuracies/len(y_true_list)


def plot_accuracy_curves(approx_y_true,approx_y_pred,topx_y_true,topx_y_pred,classes,plot=False):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    results = {"approximate_accuracy": [],
               "topx_accuracy": [],
               "mean_percentage": []
    }
    start=1
    end=10
    for x in range(start,end):
        approx_acc=approximate_accuracy(approx_y_true,approx_y_pred,classes,x-1)
        mean_percentage,topx_acc=topx_accuracy(topx_y_true,topx_y_pred,x)
        results['approximate_accuracy'].append(approx_acc)
        results['mean_percentage'].append(mean_percentage)
        results['topx_accuracy'].append(topx_acc)
    # Get the loss values of the results dictionary (training and test)
    approx_acc = results['approximate_accuracy']
    mean_percentage = results['mean_percentage']
    topx_acc = results['topx_accuracy']
    

    # Figure out how many epochs there were
    
    accuracy_amount = range(start,end)
    if(plot):
        # Setup a plot 
        plt.figure(figsize=(15, 7))

        # Plot loss
        #plt.subplot(1, 2, 1)
        plt.plot(accuracy_amount, approx_acc, label='Approximate Accuracy')
        plt.plot(accuracy_amount, topx_acc, label='Top Values Accuracy')
        # plt.plot(accuracy_amount, mean_percentage, label='mean_percentage')
        plt.title('Accuracy')
        plt.xlabel('Accuracy Distance')
        plt.legend()
        plt.show()
    return results






if __name__ == "__main__":
    main()