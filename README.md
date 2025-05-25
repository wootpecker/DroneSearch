# DroneSearch

DroneSearch is a machine learning project for analyzing and predicting simulation data for implementation in real-world experiments using a nano-drone.
The repository includes scripts for data preprocessing, model building, training, evaluation, and visualization.

## Features

- Main training class ([train_model.py](train_model.py))
- Prediction and evaluation ([predictions.py](predictions.py))
- Data transformation and augmentation ([data_transformations.py](data_transformations.py), [data_transformations_with_adequate_input.py](data_transformations_with_adequate_input.py))
- Dataset creation ([create_dataset.py](create_dataset.py))
- Model building ([model_builder.py](model_builder.py))
- Custom data loaders ([model_dataloader.py](model_dataloader.py))
- Engine for training and testing step ([engine.py](engine.py), [engine_encdec.py](engine_encdec.py))
- Visualization of results ([plot_predictions.py](plot_predictions.py))
- Utility functions ([utils.py](utils.py))

## Directory Structure

- `data/` - Raw and processed datasets
- `helper/` - Helper scripts for creating dataset
- `logs/` - Training and evaluation logs
- `model/` - Saved models and checkpoints
- `plots/` - Scripts for plots of data augmentation visualization
- `results/` - Prediction results and evaluation metrics

## Getting Started

1. **Install dependencies**  <br/>
   Make sure you have Python 3.8+ installed. Install required packages:<br/>
   pip install -r requirements.txt<br/>
   My advice: install pytorch from https://pytorch.org/get-started/locally/ for enabling GPU

2. **Prepare the dataset**<br/>
    Place your raw data in the data/ directory (works with original or datasets_tensor)<br/>
    Skip to training or run:<br/>
    python create_dataset.py

3. **Train the model**<br/>
    Train model with:<br/>
    python train_model.py<br/>
    Or place CPU or GPU model in the model/ directory and skip to predictions

4. **Make predictions**
    python predictions.py
