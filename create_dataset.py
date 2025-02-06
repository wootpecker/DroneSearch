from helper import original_dataset_to_tensor,augment_dataset, train_test_split
from  logs import logger

from pathlib import Path
import os
import logging
SOURCE_DIR = 'data/original/'


def main():
    #logger.logging_config(logs_save=False)
    #create_dataset(save=False)
    pass
    
def create_dataset(amount_samples=8,window_size=[64, 64],train_ratio=0.8, save=True):
  #logger.logging_config(logs_save=False)
  initialize()
  train_GDM,train_GSL,test_GDM,test_GSL=transformation(amount_samples=amount_samples,window_size=window_size,train_ratio=train_ratio, save=save)  
  return train_GDM,train_GSL,test_GDM,test_GSL


def initialize():
  target_dir_path = Path(f"data")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"data/datasets_tensor")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  files=os.listdir(target_dir_path)
  if len(files)==0:
    logging.info("INITIALIZE")
    original_dataset_to_tensor.create_dataset_tensor(log_normalize=True, plume_threshold=10)

def transformation(amount_samples=8,window_size=[64, 64],train_ratio=0.8,save=True):
   all_datasets_GDM,all_datasets_GSL=augment_dataset.create_augmented_dataset(amount_samples=amount_samples,window_size=window_size)
   train_GDM,train_GSL,test_GDM,test_GSL=train_test_split.load_and_split_dataset(datasets_GDM=all_datasets_GDM,datasets_GSL=all_datasets_GSL,train_ratio=train_ratio,save=save,window_size=window_size) 
   return train_GDM,train_GSL,test_GDM,test_GSL

if __name__ == "__main__":
    main()
