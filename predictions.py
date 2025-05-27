"""
predictions.py

This module provides utility functions for making predictions with both machine learning models, 
specifically for evaluating and visualizing the performance of models on test datasets.
It is designed for use in the context of drone-based search applications, where models
predict spatial locations or classes from input data.
Part of this code has been implemented from: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set

-----------------------------
Testing Parameters:
- TRANSFORMED_DATASET (bool): Whether to use the transformed dataset for evaluation.
- TRANSFORMED_MODEL (bool): Whether to load the transformed or original model (only for model comparison, no need to change).
- MULTIPLE_PLOTS (bool): If True, multiple samples are plotted for each model, otherwise only one sample is plotted.
- PLOT_SETTINGS (list of str): All possible types of plots to generate ["models": model comparison, "transform": transformation comparison]
- TEST_PLOT (int): 0 for model comparison, 1 for transform comparison.
- PLOT_BA (bool): Whether to plot specific samples used for images.
- TEST_SEED (int): Random seed for reproducibility in testing.
- PLOT_1_MODEL (bool): If True, only one model is evaluated.
- PLOT_1_MODEL_TYPE (int): Model to evaluate if PLOT_1_MODEL is True, 0 for VGG8, 1 for UnetS.
- TESTING_MODEL(list) : Empty List to store models to test

Training Parameters:
- Identical to train_model.py

Hyperparameters:
- Identical to train_model.py

Constants:
- device (str): Target device for training (CPU or CUDA). [Training has been performed on GPU -> 9x times faster than CPU training]

-----------------------------
Functions:
- main(): 
    Entry point for running prediction and visualization.

- plot_models():
    Loads models, performs predictions, computes metrics, and generates plots for model comparison.

- plot_model_accuracies(result_dic):
    Generates plots for model accuracies based on predictions.

- plot_transform():
    Loads models, performs predictions, computes metrics, and generates plots for transformation comparison.

- plot_transform_accuracies(result_dic, model_type):
    Generates plots for transformation accuracies based on predictions.

- do_predictions(model_type, multiple_plots):
    Makes predictions with a specified model type, computes accuracies, and generates plots.

- make_prediction_all_results(model, test_dataloader, model_type):
    Performs predictions on the test dataset, returning predicted labels, ground truth labels, and logits.

- reshape_tensor(y_list, y_logit_list, y_preds_percent):
    Reshapes tensors for visualization and analysis.

- print_metrics(y_pred, y_list, y_preds_percent, classes, model_type):
    Computes and logs various accuracy metrics based on predictions and ground truth labels.

- topn_accuracy(y_true_list, y_predicted_list, amount_of_values):
    Computes top-N accuracy and confidence scores based on predicted and true labels.

- calculate_benchmark_accuracies(X_list, y_list, y_preds_percent):
    Computes benchmark accuracies based on maximum prediction.

- approximate_accuracy(y_true_list, y_predicted_list, height, distance):
    Computes approximate accuracy based on the distance from predicted to true labels.

- calculate_accuracies(approx_y_true, approx_y_pred, topn_y_true, topn_y_pred, classes, plot=False):
    Computes various accuracy metrics including approximate accuracy, top-N accuracy, and confidence scores.

- make_plots(y_pred, X_list, y_list, y_logit_list, y_preds_percent, multiple_plots=False):
    Generates plots for model predictions, either multiple samples or a single training sample.

- make_plots_trainingsample(y_pred, X_list, y_list, y_logit_list, y_preds_percent):
    Generates plots for a single training sample, showing input images, ground truth, model logits, and predicted outputs.

- make_plots_multiple(y_pred, X_list, y_list, y_logit_list, y_preds_percent):
    Generates multiple plots for model predictions, showing input images, ground truth, model logits, and predicted outputs for multiple samples.

-----------------------------
Dependencies:
- torch, matplotlib, tqdm, random, math, logging, pathlib, os
- Custom modules: logger.logs, model_dataloader, model_builder, utils, train_model

-----------------------------
Usage:
Run this as a script or import it as a utility in larger projects, with:
    python predictions.py
"""
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import math
import logging
from pathlib import Path
import os

from logs import logger
import model_dataloader
import model_builder
import utils
import train_model

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

HYPER_PARAMETERS = train_model.HYPER_PARAMETERS
TRAINING_PARAMETERS = train_model.TRAINING_PARAMETERS

TESTING_PARAMETERS = {              
               "TRANSFORMED_DATASET": True,               # Whether to use the transformed dataset for evaluation
               "TRANSFORMED_MODEL": True,                 # Wheter to load the transformed or original model (no need to change)          
               "MULTIPLE_PLOTS": True,                    # If True, multiple samples are plotted for each model, otherwise only one sample is plotted 
               "PLOT_SETTINGS" : ["models","transform"],  # Plot Model Comparison or Transform Comparison
               "TEST_PLOT" : 0,                           # 0: Plot Models Comparison, 1: Plot Transform Comparison
               "PLOT_BA": True,                           # If True, specific samples used for images are plotted        
               "TEST_SEED": 1009,                         # Random seed for reproducibility in testing
               "PLOT_1_MODEL": False,                     # If True, only one model is plotted
               "PLOT_1_MODEL_TYPE": 0,                    # Model to plot if PLOT_1_MODEL is True, 0: VGG8, 1: UnetS
               "TESTING_MODEL" : []
               
  }



if TESTING_PARAMETERS['PLOT_1_MODEL']:
    TESTING_PARAMETERS['TESTING_MODEL']= [TRAINING_PARAMETERS['MODEL_TYPES'][TESTING_PARAMETERS['PLOT_1_MODEL_TYPE']]]
else:
    TESTING_PARAMETERS['TESTING_MODEL']=TRAINING_PARAMETERS['MODEL_TYPES']

if TESTING_PARAMETERS['PLOT_BA']:
    HYPER_PARAMETERS['AMOUNT_SAMPLES'] = 1
    TRAINING_PARAMETERS['TEST_SEED'] = 16923



def main():
    logger.logging_config(logs_save=False)
    if TESTING_PARAMETERS['PLOT_SETTINGS'][TESTING_PARAMETERS['TEST_PLOT']]==TESTING_PARAMETERS['PLOT_SETTINGS'][0]:
        plot_models()
    else:
        plot_transform()




def plot_models():
    logging.info(f"Plotting Models")
    logging.info(f"With Models: {TESTING_PARAMETERS['TESTING_MODEL']}")
    result_dic=[]
    for model_type in TESTING_PARAMETERS['TESTING_MODEL']:
        accuracy_results=do_predictions(model_type=model_type, multiple_plots=TESTING_PARAMETERS['MULTIPLE_PLOTS'])
        for accuracy_type, accuracy_values in accuracy_results.items():
            logging.info(f"[ACCURACY] {accuracy_type}: {accuracy_values}")
        #ogging.info(f"[ACCURACY] Results: {accuracy_results}")
        result_dic.append(accuracy_results)
    plot_model_accuracies(result_dic)



def plot_model_accuracies(result_dic):
    if(TESTING_PARAMETERS['TRANSFORMED_DATASET']):
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
        label = f'{TRAINING_PARAMETERS["MODEL_TYPES"][x]}'
        plt.plot(accuracy_amount, model['approximate_accuracy'], label=label, color=f'C{x}')
    plt.plot(accuracy_amount, model['benchmark_aproximate_accuracy'], label='Benchmark', color='red')
    plt.ylabel('Accuracy', labelpad=5, fontsize=14)
    plt.xlabel('N', labelpad=5, fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.tight_layout()
    plt.savefig(target_dir_path / f'approximate_accuracy_model_comparison{figname}.pdf')
    #plt.show()

    plt.figure(figsize=(11, 5))
    for x in range(len(result_dic)):
        model=result_dic[x]
        label = f'{TRAINING_PARAMETERS["MODEL_TYPES"][x]}'
        plt.plot(accuracy_amount, model['topn_accuracy'], label=label, color=f'C{x}')
    plt.plot(accuracy_amount, model['benchmark_topn_accuracy'], label='Benchmark', color='red')        
    plt.ylabel('Accuracy', labelpad=5, fontsize=14)
    plt.xlabel('N', labelpad=5, fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.tight_layout()
    plt.savefig(target_dir_path / f'topn_accuracy_model_comparison{figname}.pdf')
    plt.show()



    plt.figure(figsize=(11, 5))
    for x in range(len(result_dic)):
        model=result_dic[x]
        label = f'{TRAINING_PARAMETERS["MODEL_TYPES"][x]}'
        plt.plot(accuracy_amount, model['confidence'], label=label, color=f'C{x}')
    plt.ylabel('Mean Percentage', labelpad=5, fontsize=14)
    plt.xlabel('N', labelpad=5, fontsize=14)    
    plt.legend(loc='best', fontsize=14)
    plt.tight_layout()
    plt.savefig(target_dir_path / f'confidence_model_comparison{figname}.pdf')

    #plt.show()    
    return result_dic



def plot_transform():
    logging.info(f"Plotting Transform Comparison")
    logging.info(f"With Models: {TESTING_PARAMETERS['TESTING_MODEL']}")
    for model_type in TESTING_PARAMETERS['TESTING_MODEL']:
        result_dic=[]
        #global TESTING_PARAMETERS
        TESTING_PARAMETERS['TRANSFORMED_MODEL']=True
        for i in range(0,2):        
            accuracy_results=do_predictions(model_type=model_type, multiple_plots=TESTING_PARAMETERS['MULTIPLE_PLOTS'])
            for accuracy_type, accuracy_values in accuracy_results.items():
                logging.info(f"[ACCURACY] {accuracy_type}: {accuracy_values}")
            result_dic.append(accuracy_results)
            TESTING_PARAMETERS['TRANSFORMED_MODEL']=False
        plot_transform_accuracies(result_dic, model_type)




def plot_transform_accuracies(result_dic,model_type=TRAINING_PARAMETERS['MODEL_TYPES'][0]):
    if(TESTING_PARAMETERS['TRANSFORMED_DATASET']):
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

    if(model_type==TRAINING_PARAMETERS['MODEL_TYPES'][0]):
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
    plt.savefig(target_dir_path / f'approximate_accuracy_{model_type}{figname}.pdf')
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
    plt.savefig(target_dir_path / f'evaluation_{model_type}{figname}.pdf')
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
    plt.savefig(target_dir_path / f'confidence_{model_type}{figname}.pdf')

    #plt.show()    
    return result_dic





def do_predictions(model_type= "VGG", multiple_plots=False):
    """Makes prediction with a model and plots 5 different test samples with respective results.
    

    Args:
    model(string): Type of model to be used for predicting.
    dataloader(string): Type of dataset to be used for predicting.

    Returns:
    Plot of 5 samples
    Plot of confusion matrix
    """
    utils.seed_generator(SEED=TRAINING_PARAMETERS['LOAD_SEED'])
    train_dataloader,test_dataloader,classes = model_dataloader.create_dataloader(model_type=model_type, batch_size=HYPER_PARAMETERS['BATCH_SIZE'], transform=TESTING_PARAMETERS['TRANSFORMED_DATASET'], amount_samples=HYPER_PARAMETERS['AMOUNT_SAMPLES'], window_size=HYPER_PARAMETERS['WINDOW_SIZE'])
    model = model_builder.choose_model(model_type=model_type,output_shape=classes,device=device,window_size=HYPER_PARAMETERS['WINDOW_SIZE'])
    model,_=utils.load_model(model= model, model_type=model_type, device=device, transform=TESTING_PARAMETERS['TRANSFORMED_MODEL'])
    
    utils.seed_generator(SEED=TRAINING_PARAMETERS['TEST_SEED'])
    y_pred,y_list,X_list,y_logit_list,y_preds_percent=make_prediction_all_results(model_type=model_type,model=model,test_dataloader=test_dataloader)
    accuracy_results=print_metrics(y_pred,y_list,y_preds_percent,classes,model_type)
    benchmark_similarity,benchmark_aproximate_accuracy,benchmark_topn_accuracy,benchmark_euclidean_distance=calculate_benchmark_accuracies(X_list,y_list,y_preds_percent)
    accuracy_results['benchmark_aproximate_accuracy']=benchmark_aproximate_accuracy
    accuracy_results['benchmark_similarity']=benchmark_similarity
    accuracy_results['benchmark_topn_accuracy']=benchmark_topn_accuracy
    accuracy_results['benchmark_euclidean_distance']=benchmark_euclidean_distance
    y_list,y_logit_list,y_preds_percent=reshape_tensor(y_list=y_list,y_logit_list=y_logit_list,y_preds_percent=y_preds_percent)
    make_plots(y_pred,X_list,y_list,y_logit_list,y_preds_percent, multiple_plots=multiple_plots)  
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
        if(model_type==TRAINING_PARAMETERS['MODEL_TYPES'][1]):
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
    if(model_type==TRAINING_PARAMETERS['MODEL_TYPES'][1]):
        y_pred_index = torch.argmax(y_preds_percent.view(y_preds_percent.shape[0], -1), dim=1)  # [batch]
        y_list_index = torch.argmax(y_list.view(y_list.shape[0], -1), dim=1)  # [batch]

        return calculate_accuracies(y_list_index,y_pred_index,y_list,y_preds_percent,y_pred.shape[-1])

    y_old=y_list            
    y_list=torch.argmax(y_list,dim=1)
    return calculate_accuracies(y_list,y_pred,y_old,y_preds_percent,math.sqrt(classes))




def topn_accuracy(y_true_list, y_predicted_list, amount_of_values):
    topn_values, topn_indices = torch.topk(y_predicted_list.view(y_predicted_list.shape[0], -1), amount_of_values, dim=1)
    y_true_max, y_true_indices = torch.max(y_true_list.view(y_true_list.shape[0], -1), dim=1)
    matches= torch.any(y_true_indices[:,None] == topn_indices, dim=1)
    accuracy = matches.float().mean().item()
    mean_percentage=torch.mean(topn_values[:,amount_of_values-1].float())
    return mean_percentage.float().item(),accuracy



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
    benchmark_euclidean_distance=average_euclidean_distance(y_index,x_max_index,X_list.shape[-1])
    return accuracy_max_pred,approx_accuracies,topn_accuracies,benchmark_euclidean_distance


def average_euclidean_distance(y_true_list, y_predicted_list,height):
    distances = []
    for y_true, y_predicted in zip(y_true_list, y_predicted_list):
        y_true_height = torch.div(y_true, height, rounding_mode='floor')
        y_true_width = y_true % height
        y_predicted_height = torch.div(y_predicted, height, rounding_mode='floor')
        y_predicted_width = y_predicted % height

        distance= torch.sqrt((y_true_height - y_predicted_height) ** 2 + (y_true_width - y_predicted_width) ** 2)
        distances.append(distance)
    average_distance = torch.mean(torch.stack(distances))
    return average_distance.item()


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

        # Count the number of matches
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
               "euclidean_distance": [],
               "topn_accuracy": [],
               "confidence": [],
               "benchmark_aproximate_accuracy": [],
               "benchmark_similarity": [],
               "benchmark_topn_accuracy": [],
               "benchmark_euclidean_distance": []
               }
 
    start=1
    end=10
    for x in range(start,end):
        approx_acc=approximate_accuracy(approx_y_true,approx_y_pred,classes,x-1)
        confidence,topn_acc=topn_accuracy(topn_y_true,topn_y_pred,x)
        results['approximate_accuracy'].append(approx_acc)
        results['confidence'].append(confidence)
        results['topn_accuracy'].append(topn_acc)

    euclidean_distance=average_euclidean_distance(approx_y_true,approx_y_pred,classes)
    results['euclidean_distance'].append(euclidean_distance)
    return results



def make_plots(y_pred,X_list,y_list,y_logit_list,y_preds_percent, multiple_plots=False):
    if multiple_plots:
        make_plots_multiple(y_pred,X_list,y_list,y_logit_list,y_preds_percent)
    else:
        make_plots_trainingsample(y_pred,X_list,y_list,y_logit_list,y_preds_percent)
        #make_trainingsample_plot2x2(y_pred,X_list,y_list,y_logit_list,y_preds_percent)

def make_plots_trainingsample(y_pred,X_list,y_list,y_logit_list,y_preds_percent):
    y_list,y_logit_list,y_preds_percent=reshape_tensor(y_list,y_logit_list,y_preds_percent)
    size=[1,4]
    fig_width = 4 * 4  # 4 columns × 2 inches per image
    fig_height = 1 * 4  # 5 rows × 2 inches per image
    f, arr = plt.subplots(size[0], size[1], figsize=(fig_width, fig_height))#, tight_layout=True)
    random_samples=random.sample(range(X_list.shape[0]),k=5)
    for j in range(arr.shape[0]):
        i=random_samples[j]
        arr[0].imshow(X_list[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")        
        arr[1].imshow(y_list[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")
        y_target,x_target=divmod(y_list[i].argmax().item(),X_list[i].shape[-1])
        arr[2].imshow(y_logit_list[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")
        y_pred,x_pred=divmod(y_preds_percent[i].argmax().item(),X_list[i].shape[-1])   
        arr[3].imshow(y_preds_percent[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")        
        if(y_target==y_pred and x_target==x_pred):
            color='green'
        else:
            color='red'     

        arr[j].set_xlabel(f"x (dm)", fontsize=14)
        arr[j].set_ylabel(f"y (dm)", fontsize=14)
        arr[j].label_outer()        
        if(j==0):
            arr[0].set_title(f"Input (X)", fontsize=18)
            arr[1].set_title(f"Target (y)", fontsize=18)
            arr[2].set_title(f"Output of Model (y_logit)", fontsize=18)
            arr[3].set_title(f"Prediction of Model", fontsize=18)

        for k in range(4):
            arr[k].set_xticks(range(0, X_list[i].shape[-1], 10))
            arr[k].set_yticks(range(0, X_list[i].shape[-1], 10))
            arr[k].tick_params(axis='x', labelsize=14)
            arr[k].tick_params(axis='y', labelsize=14)

    output_dir = Path("results/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "training_sample.pdf")
    plt.show()



def make_plots_multiple(y_pred,X_list,y_list,y_logit_list,y_preds_percent):
    y_list,y_logit_list,y_preds_percent=reshape_tensor(y_list,y_logit_list,y_preds_percent)
    size=[3,3]
    fig_width = 3 * 3  # 4 columns × 2 inches per image
    fig_height = 3 * 3  # 5 rows × 2 inches per image
    f, arr = plt.subplots(size[0],size[1], figsize=(fig_width, fig_height)) 
    random_samples=random.sample(range(X_list.shape[0]),k=20)
    for j in range(arr.shape[0]):
        i=random_samples[j+5]
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
            arr[j,0].set_title(f"Inputs, Max at ({x_target},{y_target})", fontsize=13)
            arr[j,1].set_title(f"Outputs", fontsize=13)
            arr[j,2].set_title(f"Predictions, Max={y_preds_percent[i].max():.2f} at ({x_pred},{y_pred})", color=color, fontsize=13)

        else:
            arr[j,0].set_title(f"Max at ({x_target},{y_target})", fontsize=13)
            arr[j,2].set_title(f"Max={y_preds_percent[i].max():.2f} at ({x_pred},{y_pred})", color=color, fontsize=13)
        for k in range(3):
            arr[j, k].set_xlabel("x (dm)", fontsize=12)
            arr[j, k].set_ylabel("y (dm)", fontsize=12)
            arr[j, k].set_xticks(range(0, X_list[i].shape[-1], 10))
            arr[j, k].set_yticks(range(0, X_list[i].shape[-1], 10))
            arr[j, k].set_aspect('equal')

        arr[j,0].plot(x_target, y_target, marker="*", markersize=10, markeredgecolor="red", markerfacecolor="red") 
        
        for ax in arr.flat:
            ax.label_outer()

    f.tight_layout()
    plt.tight_layout()
    
    output_dir = Path("results/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    files=os.listdir(output_dir)
    plt.savefig(output_dir / f"training_sample_multiple_{len(files)+1:03d}.pdf")    
    plt.show()




# Just for visualtion purposes, not used in the code
def make_trainingsample_plot2x2(y_pred, X_list, y_list, y_logit_list, y_preds_percent):
    y_list, y_logit_list, y_preds_percent = reshape_tensor(y_list, y_logit_list, y_preds_percent)
    size = [2, 2]
    fig_width = 2 * 4  # 2 columns × 4 inches per image
    fig_height = 2 * 4  # 2 rows × 4 inches per image
    f, arr = plt.subplots(size[0], size[1], figsize=(fig_width, fig_height))
    random_samples = random.sample(range(X_list.shape[0]), k=5)
    for j in range(len(random_samples)):
        i = random_samples[j]
        row, col = divmod(j, 2)
        if j == 0:
            arr[row, col].imshow(X_list[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")
            arr[row, col].set_title("Input (X)", fontsize=18)
        elif j == 1:
            arr[row, col].imshow(y_list[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")
            arr[row, col].set_title("Target (y)", fontsize=18)
        elif j == 2:
            arr[row, col].imshow(y_logit_list[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")
            arr[row, col].set_title("Output of Model (y_logit)", fontsize=18)
        elif j == 3:
            arr[row, col].imshow(y_preds_percent[i].squeeze(0).unsqueeze(-1).numpy(), origin="lower")
            arr[row, col].set_title("Prediction of Model", fontsize=18)
            y_target, x_target = divmod(y_list[i].argmax().item(), X_list[i].shape[-1])
            y_pred_val, x_pred = divmod(y_preds_percent[i].argmax().item(), X_list[i].shape[-1])
            color = 'green' if (y_target == y_pred_val and x_target == x_pred) else 'red'
            arr[row, col].plot(x_target, y_target, marker="*", markersize=10, markeredgecolor=color, markerfacecolor=color)
        arr[row, col].set_xlabel("x (dm)", fontsize=14)
        arr[row, col].set_ylabel("y (dm)", fontsize=14)
        arr[row, col].set_xticks(range(0, X_list[i].shape[-1], 10))
        arr[row, col].set_yticks(range(0, X_list[i].shape[-1], 10))
        arr[row, col].tick_params(axis='x', labelsize=14)
        arr[row, col].tick_params(axis='y', labelsize=14)
        arr[row, col].label_outer()
    if len(random_samples) < 4:
        for j in range(len(random_samples), 4):
            row, col = divmod(j, 2)
            f.delaxes(arr[row, col])
    plt.tight_layout()
    output_dir = Path("results/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "training_sample_2x2.pdf")
    plt.show()







if __name__ == "__main__":
    main()