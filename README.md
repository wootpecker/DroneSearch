# DroneSearch

DroneSearch is a machine learning project for analyzing and predicting simulation data for implementation in real-world experiments using a nano-drone. 
The repository includes scripts for data preprocessing, model building, training, evaluation, and visualization.

## Features

- Data transformation and augmentation ([data_transformations.py](data_transformations.py), [data_transformations_with_adequate_input.py](data_transformations_with_adequate_input.py))
- Dataset creation ([create_dataset.py](create_dataset.py))
- Model building ([model_builder.py](model_builder.py))
- Custom data loaders ([model_dataloader.py](model_dataloader.py))
- Engine for training and testing step ([engine.py](engine.py), [engine_encdec.py](engine_encdec.py))
- Main training class ([train_model.py](train_model.py))
- Prediction and evaluation ([predictions.py](predictions.py))
- Visualization of results ([plot_predictions.py](plot_predictions.py))
- Utility functions ([utils.py](utils.py))

## Directory Structure

- `data/` - Raw and processed datasets
- `helper/` - Helper scripts and utilities
- `logs/` - Training and evaluation logs
- `model/` - Saved models and checkpoints
- `plots/` - Plots for visualizations
- `results/` - Prediction results and evaluation metrics

## Getting Started

1. **Install dependencies**  
   Make sure you have Python 3.8+ installed. Install required packages:
   pip install -r requirements.txt

2. **Prepare the dataset**
    Place your raw data in the data/ directory and run:
    python create_dataset.py
    or skip to directly to training

3. **Train the model**
    python train_model.py

4. **Make predictions**
    python predictions.py
