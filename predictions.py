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
import ds4_train_model
from pathlib import Path
import os
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPES = ["VGG8", "UnetS", "VGGVariation"]
TRANSFORMED_MODEL=True
HYPER_PARAMETERS = ds4_train_model.HYPER_PARAMETERS
TRAINING_PARAMETERS = ds4_train_model.TRAINING_PARAMETERS



HYPER_PARAMETERS['AMOUNT_SAMPLES'] = 1
HYPER_PARAMETERS['TRANSFORM'] = True



MODEL_TO_TEST=[HYPER_PARAMETERS['MODEL_TYPES'][0],HYPER_PARAMETERS['MODEL_TYPES'][1]]
#MODEL_TO_TEST=[HYPER_PARAMETERS['MODEL_TYPES'][1]]


def main():
    
    logger.logging_config(logs_save=False)
    #plot_models()
    for models in MODEL_TO_TEST:
        plot_transform(model_type=models)

    #plot_transform(model_type=HYPER_PARAMETERS['MODEL_TYPES'][1])


def plot_models():
    result_dic=[]
    for model_type in MODEL_TO_TEST:
        accuracy_results=do_predictions(model_type=model_type)
        logging.info(f"[ACCURACY] Results: {accuracy_results}")
        #print(accuracy_results)
        result_dic.append(accuracy_results)
    plot_model_accuracies(result_dic)



def plot_model_accuracies(result_dic):
    if(HYPER_PARAMETERS['TRANSFORM']):
        figname='_on_transformed_data'
    else:
        figname='_on_original_data'    
    target_dir_path = Path(f"results")
    target_dir_path.mkdir(parents=True, exist_ok=True)
    target_dir_path = Path(f"results/accuracies/")
    target_dir_path.mkdir(parents=True, exist_ok=True)
    target_dir_path = Path(f"results/accuracies/model_comparison/")
    target_dir_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 5))

    accuracy_amount =  range(1,len(result_dic[0]['approximate_accuracy'])+1)
    for x in range(len(result_dic)):
        model=result_dic[x]
        plt.plot(accuracy_amount, model['approximate_accuracy'], label=f'{MODEL_TO_TEST[x]}', color=f'C{x}')
    plt.plot(accuracy_amount, model['benchmark_aproximate_accuracy'], label='Benchmark', color='red')
    # plt.plot(accuracy_amount, confidence, label='confidence')
    plt.title('Approximate Accuracy', pad=10)
    plt.ylabel('Accuracy', labelpad=5)
    plt.xlabel('N', labelpad=5)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(target_dir_path / f'approximate_accuracy_model_comparison{figname}.png')
    #plt.show()

    plt.figure(figsize=(11, 5))
    for x in range(len(result_dic)):
        model=result_dic[x]
        plt.plot(accuracy_amount, model['topn_accuracy'], label=f'{MODEL_TO_TEST[x]}', color=f'C{x}')
    plt.plot(accuracy_amount, model['benchmark_topn_accuracy'], label='Benchmark', color='red')        
    # plt.plot(accuracy_amount, confidence, label='confidence')
    plt.title('Top N Accuracy', pad=10)
    plt.ylabel('Accuracy', labelpad=5)
    plt.xlabel('N', labelpad=5)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(target_dir_path / f'topn_accuracy_model_comparison{figname}.png')
    #plt.show()


    plt.figure(figsize=(11, 5))
    for x in range(len(result_dic)):
        model=result_dic[x]
        plt.plot(accuracy_amount, model['confidence'], label=f'{MODEL_TO_TEST[x]}', color=f'C{x}')
        #plt.plot(accuracy_amount, model['topn_accuracy'], label=f'{model["model_type"]} Top Values Accuracy')
    # plt.plot(accuracy_amount, confidence, label='confidence')
    plt.title('Confidence Comparison', pad=10)
    plt.ylabel('Mean Percentage', labelpad=5)
    plt.xlabel('N', labelpad=5)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(target_dir_path / f'confidence_model_comparison{figname}.png')

    #plt.show()    
    return result_dic



def plot_transform(model_type=HYPER_PARAMETERS['MODEL_TYPES'][0]):

    result_dic=[]
    global TRANSFORMED_MODEL
    TRANSFORMED_MODEL=True
    for i in range(0,2):        
        accuracy_results=do_predictions(model_type=model_type)
        logging.info(f"[ACCURACY] Results: {accuracy_results}")
        #print(accuracy_results)
        result_dic.append(accuracy_results)
        TRANSFORMED_MODEL=False
    plot_transform_accuracies(result_dic, model_type)




def plot_transform_accuracies(result_dic,model_type=HYPER_PARAMETERS['MODEL_TYPES'][0]):
    if(HYPER_PARAMETERS['TRANSFORM']):
        figname='_on_transformed_data'
        labelname='Evaluation on transformed data'        
    else:
        figname='_on_original_data'
        labelname='Evaluation on original data'
    target_dir_path = Path(f"results")
    target_dir_path.mkdir(parents=True, exist_ok=True)
    target_dir_path = Path(f"results/accuracies")
    target_dir_path.mkdir(parents=True, exist_ok=True)
    target_dir_path = Path(f"results/accuracies/transform_comparison/")
    target_dir_path.mkdir(parents=True, exist_ok=True)
    label = [f'{model_type}', f'{model_type} without Transformation']

    if(model_type==MODEL_TYPES[0]):
        color_bias = 0
        factor = 2
    else:
        color_bias = 1
        factor = 3
    plt.figure(figsize=(11, 5))
    accuracy_amount =  range(1,len(result_dic[0]['approximate_accuracy'])+1)
    
    for x in range(len(result_dic)):
        model=result_dic[x]
        plt.plot(accuracy_amount, model['approximate_accuracy'], label=label[x], color=f'C{color_bias+factor*x}')
    plt.plot(accuracy_amount, model['benchmark_aproximate_accuracy'], label='Benchmark', color='red')    
    plt.title('Approximate Accuracy', pad=10)
    plt.ylabel('Accuracy', labelpad=5)
    plt.xlabel('N', labelpad=5)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(target_dir_path / f'approximate_accuracy_{model_type}{figname}.png')
    #plt.show()

    plt.figure(figsize=(11, 5))
    for x in range(len(result_dic)):
        model=result_dic[x]
        plt.plot(accuracy_amount, model['approximate_accuracy'], label='Approximation Accuracy of '+label[x], color=f'C{color_bias+factor*x}')
        plt.plot(accuracy_amount, model['topn_accuracy'], label='Top N Accuracy of '+label[x], color=f'C{color_bias+factor*x}', linestyle='dashed')
    plt.plot(accuracy_amount, model['benchmark_aproximate_accuracy'], label='Approximation Accuracy of Benchmark', color='red')
    plt.plot(accuracy_amount, model['benchmark_topn_accuracy'], label='Top N Accuracy of Benchmark', color='red', linestyle='dashed')
    plt.title(f'{labelname}', pad=10)
    plt.ylabel('Accuracy', labelpad=5)
    plt.xlabel('N', labelpad=5)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(target_dir_path / f'evaluation_{model_type}{figname}.png')
    #plt.show()


    plt.figure(figsize=(11, 5))
    for x in range(len(result_dic)):
        model=result_dic[x]     
        plt.plot(accuracy_amount, model['confidence'], label=label[x], color=f'C{color_bias+factor*x}')

    plt.title('Confidence Comparison', pad=10)
    plt.ylabel('Mean Percentage', labelpad=5)
    plt.xlabel('N', labelpad=5)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(target_dir_path / f'confidence_{model_type}{figname}.png')

    #plt.show()    
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
    model,_=utils.load_model(model= model, model_type=model_type, device=device, transform=TRANSFORMED_MODEL)
    
    utils.seed_generator(SEED=TRAINING_PARAMETERS['LOAD_SEED'])
    y_pred,y_list,X_list,y_logit_list,y_preds_percent=make_prediction_all_results(model_type=model_type,model=model,test_dataloader=test_dataloader)
    accuracy_results=print_metrics(y_pred,y_list,y_preds_percent,classes,model_type)
    benchmark_similarity,benchmark_aproximate_accuracy,benchmark_topn_accuracy=calculate_benchmark_accuracies(X_list,y_list,y_preds_percent)
    accuracy_results['benchmark_aproximate_accuracy']=benchmark_aproximate_accuracy
    accuracy_results['benchmark_similarity']=benchmark_similarity
    accuracy_results['benchmark_topn_accuracy']=benchmark_topn_accuracy
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
    







def reshape_tensor(y_list,y_logit_list,y_preds_percent):
    if(len(y_list.shape)>2):
        return y_list,y_logit_list,y_preds_percent
    else:
        y_list=torch.reshape(y_list, (y_list.shape[0], HYPER_PARAMETERS['WINDOW_SIZE'][0], HYPER_PARAMETERS['WINDOW_SIZE'][1]))
        y_logit_list=torch.reshape(y_logit_list, (y_logit_list.shape[0], HYPER_PARAMETERS['WINDOW_SIZE'][0], HYPER_PARAMETERS['WINDOW_SIZE'][1]))
        y_preds_percent=torch.reshape(y_preds_percent, (y_preds_percent.shape[0], HYPER_PARAMETERS['WINDOW_SIZE'][0], HYPER_PARAMETERS['WINDOW_SIZE'][1]))        
    return y_list,y_logit_list,y_preds_percent






def print_metrics(y_pred,y_list,y_preds_percent,classes,model_type):
    logging.info(f"Calculating Metrics and Plotting:")
    if(model_type==MODEL_TYPES[1]):
        y_pred_index = torch.argmax(y_preds_percent.view(y_preds_percent.shape[0], -1), dim=1)  # [batch]
        y_list_index = torch.argmax(y_list.view(y_list.shape[0], -1), dim=1)  # [batch]

        return calculate_accuracies(y_list_index,y_pred_index,y_list,y_preds_percent,y_pred.shape[-1])

    y_old=y_list            
    y_list=torch.argmax(y_list,dim=1)
    return calculate_accuracies(y_list,y_pred,y_old,y_preds_percent,math.sqrt(classes))






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


def test_topn():
    y_true_list = torch.randint(0, 20, (10, 3, 3))
    y_predicted_list = torch.randint(0, 20, (10, 3, 3))
    confidence,topn_accuracies=topn_accuracy(y_true_list,y_predicted_list,2)
    print("topn_accuracy",topn_accuracies)
    print("confidence",confidence)

def topn_accuracy(y_true_list, y_predicted_list, amount_of_values):
    topn_values, topn_indices = torch.topk(y_predicted_list.view(y_predicted_list.shape[0], -1), amount_of_values, dim=1)
    y_true_max, y_true_indices = torch.max(y_true_list.view(y_true_list.shape[0], -1), dim=1)
    matches= torch.any(y_true_indices[:,None] == topn_indices, dim=1)
    accuracy = matches.float().mean().item()
    mean_percentage=torch.mean(topn_values[:,amount_of_values-1].float())
    return mean_percentage.float().item(),accuracy

#test_topn()


def calculate_benchmark_accuracies(X_list,y_list,y_preds_percent):
    accuracy_max_pred=0
    x_max_index = torch.argmax(X_list.view(X_list.shape[0], -1), dim=1)  # [batch]
    y_pred_index = torch.argmax(y_preds_percent.view(y_preds_percent.shape[0], -1), dim=1) 
    y_index = torch.argmax(y_list.view(y_list.shape[0], -1), dim=1) 
    for x in range(x_max_index.shape[0]):
        if(x_max_index[x]==y_pred_index[x]):
            accuracy_max_pred+=1

    accuracy_max_pred=accuracy_max_pred/len(X_list)

    approx_accuracies=[]
    topn_accuracies=[]




    for x in range(1,10):
        approx_acc=approximate_accuracy(y_index,x_max_index,X_list.shape[-1],x-1)
        approx_accuracies.append(approx_acc)
        confidence,topn_acc=topn_accuracy(y_list,X_list,x)
        topn_accuracies.append(topn_acc)
        #print(f"approx_acc: {approx_acc}")
    #print(f"approx_acc: {approx_acc}")
    return accuracy_max_pred,approx_accuracies,topn_accuracies

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
    
    accuracies=accuracies/len(y_true_list)
    accuracies=accuracies.float().item()
    return accuracies


def calculate_accuracies(approx_y_true,approx_y_pred,topn_y_true,topn_y_pred,classes,plot=False):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    results = {"approximate_accuracy": [],
               "topn_accuracy": [],
               "confidence": [],
               "benchmark_aproximate_accuracy": [],
               "benchmark_similarity": [],
               "benchmark_topn_accuracy": []}
 
    start=1
    end=10
    for x in range(start,end):
        approx_acc=approximate_accuracy(approx_y_true,approx_y_pred,classes,x-1)
        confidence,topn_acc=topn_accuracy(topn_y_true,topn_y_pred,x)
        results['approximate_accuracy'].append(approx_acc)
        results['confidence'].append(confidence)
        results['topn_accuracy'].append(topn_acc)
    return results



def make_plots(y_pred,X_list,y_list,y_logit_list,y_preds_percent):
    #make_trainingsample_plot(y_pred,X_list,y_list,y_logit_list,y_preds_percent)
    #make_trainingsample_plot(y_pred,X_list,y_list,y_logit_list,y_preds_percent)
    make_plots_multiple(y_pred,X_list,y_list,y_logit_list,y_preds_percent)

def make_trainingsample_plot(y_pred,X_list,y_list,y_logit_list,y_preds_percent):
    #def show_as_image_sequence_batch(dataset, predicted_dataset):
    #results=[X_list,y_list,y_logit_list,y_preds_percent]
    y_list,y_logit_list,y_preds_percent=reshape_tensor(y_list,y_logit_list,y_preds_percent)
    size=[1,4]
    fig_width = 4 * 4  # 4 columns × 2 inches per image
    fig_height = 1 * 4  # 5 rows × 2 inches per image
    f, arr = plt.subplots(size[0],size[1], figsize=(fig_width, fig_height)) 
    random_samples=random.sample(range(X_list.shape[0]),k=5)
    for j in range(arr.shape[0]):
        i=random_samples[j]
        #print(i)
        arr[0].imshow(X_list[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")        
        arr[1].imshow(y_list[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")
        y_target,x_target=divmod(y_list[i].argmax().item(),X_list[i].shape[-1])
        arr[2].imshow(y_logit_list[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")
        #y_preds_percent[i]=torch.sigmoid(y_logit_list[i])
        y_pred,x_pred=divmod(y_preds_percent[i].argmax().item(),X_list[i].shape[-1])   
        arr[3].imshow(y_preds_percent[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")        
        if(y_target==y_pred and x_target==x_pred):
            color='green'
        else:
            color='red'     

        arr[j].set_xlabel(f"x (dm)")
        arr[j].set_ylabel(f"y (dm)")
        arr[j].label_outer()        
        if(j==0):
            arr[0].set_title(f"Input (X)")
            arr[1].set_title(f"Target (Y)")
            arr[2].set_title(f"Output of Model (y_logit)")
            arr[3].set_title(f"Prediction of Model")

    plt.title('Top N Accuracy', pad=10)
    plt.tight_layout()
    f.subplots_adjust(hspace =0.4)
   # f.tight_layout()
    #plt.subplots_adjust(hspace =0.4)

    output_dir = Path("results/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "training_sample.png")
    #plt.show()



def make_plots_multiple(y_pred,X_list,y_list,y_logit_list,y_preds_percent):
    #def show_as_image_sequence_batch(dataset, predicted_dataset):
    #results=[X_list,y_list,y_logit_list,y_preds_percent]
    y_list,y_logit_list,y_preds_percent=reshape_tensor(y_list,y_logit_list,y_preds_percent)
    size=[3,3]
    fig_width = 3 * 3  # 4 columns × 2 inches per image
    fig_height = 3 * 3  # 5 rows × 2 inches per image
    f, arr = plt.subplots(size[0],size[1], figsize=(fig_width, fig_height)) 
    random_samples=random.sample(range(X_list.shape[0]),k=20)
    for j in range(arr.shape[0]):
        i=random_samples[j+5]
        #print(i)
        arr[j,0].imshow(X_list[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")        
        y_target,x_target=divmod(y_list[i].argmax().item(),X_list[i].shape[-1])
        arr[j,1].imshow(y_logit_list[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")
        y_pred,x_pred=divmod(y_preds_percent[i].argmax().item(),X_list[i].shape[-1])   
        arr[j,2].imshow(y_preds_percent[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")        
        if(y_target==y_pred and x_target==x_pred):
            color='green'
        else:
            color='red'     

        
        if(j==0):
            arr[j,0].set_title(f"Inputs, Max at ({x_target},{y_target})")
            #arr[j,0].set(xlabel=f"x (dm)")
            #arr[j,0].set(ylabel=f"y (dm)")
            arr[j,1].set_title(f"Outputs")
            arr[j,2].set_title(f"Predictions, Max={y_preds_percent[i].max():.2f} at ({x_pred},{y_pred})", color=color)

        else:
            arr[j,0].set_title(f"Max at ({x_target},{y_target})")
            arr[j,2].set_title(f"Max={y_preds_percent[i].max():.2f} at ({x_pred},{y_pred})", color=color)
        for k in range(3):
            arr[j, k].set(xlabel="x (dm)", ylabel="y (dm)")
            arr[j, k].set_xticks(range(0, X_list[i].shape[-1], 10))
            arr[j, k].set_yticks(range(0, X_list[i].shape[-1], 10))
            arr[j, k].set_aspect('equal')
        #arr[j,2].set(ylabel=f"y (dm)")            
        if(j==2):
            arr[j,0].set(xlabel=f"x (dm)")
            arr[j,1].set(xlabel=f"x (dm)")
            arr[j,2].set(xlabel=f"x (dm)")


       # else:
            #arr[j,0].set_xticks([]) 
            #arr[j,0].set_yticks([]) 
        #arr[j,1].set_xlabel(f"Max Value at ({x_target},{y_target})")
                     
        #arr[j,2].set_xlabel(f"Sample:{i}") 
        arr[j,0].plot(x_target, y_target, marker="*", markersize=10, markeredgecolor="red", markerfacecolor="red") 
        
        for ax in arr.flat:
            ax.label_outer()
        #arr[j,0].plot(10, 10, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")              
        #plt.xticks([])
        #plt.yticks([])


    #f.subplots_adjust(hspace =0.4)
    f.tight_layout()
   # f.tight_layout()
    #plt.subplots_adjust(hspace =0.4)
    #plt.title('Top N Accuracy', pad=10)
    plt.tight_layout()
    
    output_dir = Path("results/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    files=os.listdir(output_dir)
    plt.savefig(output_dir / f"training_sample_multiple_{len(files)+1:03d}.png")    
    #plt.show()

          # Ensure the same scale for x and y axes











if __name__ == "__main__":
    main()