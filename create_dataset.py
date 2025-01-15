from helper import original_dataset_to_tensor,augment_dataset, train_test_split


from pathlib import Path
import os
SOURCE_DIR = 'data/original/'


def main():
    amount_samples=8
    window_size=[64, 64]
    train_ratio=0.8
    # Create dataset tensor with specified parameters
    initialize()
    transformation(amount_samples=amount_samples,window_size=window_size,train_ratio=train_ratio)
    


def initialize():
  target_dir_path = Path(f"data")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"data/datasets_tensor")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  files=os.listdir(target_dir_path)
  if len(files)==0:
    original_dataset_to_tensor.create_dataset_tensor(log_normalize=True, plume_threshold=10)

def transformation(amount_samples=8,window_size=[64, 64],train_ratio=0.8):
   all_datasets_GDM,all_datasets_GSL=augment_dataset.create_augmented_dataset(amount_samples=amount_samples,window_size=window_size)
   train_test_split.load_and_split_dataset(datasets_GDM=all_datasets_GDM,datasets_GSL=all_datasets_GSL,train_ratio=train_ratio) 
   

if __name__ == "__main__":
    main()
