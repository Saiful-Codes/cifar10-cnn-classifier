# cifar10-cnn-classifier

Convolutional Neural Network implementation for CIFAR-10 image classification using PyTorch, with hyperparameter tuning, structured training pipeline, evaluation utilities, and a deployable Streamlit application.

## Project Overview

This project implements a custom Convolutional Neural Network (CNN) from scratch to classify images from the CIFAR-10 dataset. The goal was to move beyond notebook-based experimentation and structure the project in a clean, modular format suitable for production-style development and portfolio presentation.

The project includes:

* A modular PyTorch training pipeline
* Hyperparameter tuning (learning rate, batch size, dropout)
* Model selection based on validation performance
* Final evaluation on the test set
* Confusion matrix and classification report generation
* A deployable Streamlit web application for inference

## Dataset

The CIFAR-10 dataset consists of 60,000 color images (32x32) across 10 classes:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

The dataset is automatically downloaded via torchvision.

Data preprocessing includes:

* Random horizontal flip (training only)
* Random crop with padding (training only)
* Normalization using dataset-specific mean and standard deviation
* Train/validation split with fixed random seed for reproducibility

## Model Architecture

The implemented CNN consists of:

* Conv2d (3 → 32) + BatchNorm + ReLU
* Conv2d (32 → 64) + BatchNorm + ReLU
* MaxPooling after each convolution block
* Fully connected layer (64×8×8 → 256)
* Dropout (tuned)
* Output layer (256 → 10)

CrossEntropyLoss is used for training.
Adam optimizer is used with tuned learning rates.

## Hyperparameter Tuning

A manual grid search was performed across:

* Learning rate: 1e-2, 1e-3, 1e-4
* Batch size: 32, 64
* Dropout: 0.2, 0.5

Each configuration was trained for a reduced number of epochs during tuning.
Results were recorded and sorted by best validation accuracy.

Outputs saved in the repository:

* results/tuning_results.csv
* results/best_config.json

The best configuration was selected and used for final training.

## Training Pipeline

The training process is structured into modular components:

* model.py – CNN architecture
* data.py – data loading and augmentation
* train.py – training and validation loops
* tune.py – hyperparameter grid search
* evaluate.py – test evaluation and reporting
* utils.py – helper functions

During final training:

* Training and validation losses are tracked
* Training and validation accuracy are tracked
* Best model weights are saved based on validation accuracy

Training curves are saved as:

* results/training_curves.png

## Evaluation

The final model is evaluated on the test dataset.

Metrics include:

* Test loss
* Test accuracy
* Confusion matrix
* Classification report (precision, recall, F1-score)

Saved outputs:

* results/confusion_matrix.png
* results/classification_report.txt

## Streamlit Deployment

The repository includes a Streamlit application that allows users to upload an image and receive predicted CIFAR-10 class output.

To run the app locally:

```bash
streamlit run app/app.py
```

This demonstrates model usability beyond training and showcases deployment capability.

## Project Structure

```
cifar10-cnn-classifier/
│
├── app/
├── models/
├── notebooks/
├── results/
├── src/
│   ├── model.py
│   ├── data.py
│   ├── train.py
│   ├── tune.py
│   ├── evaluate.py
│   └── utils.py
├── README.md
├── requirements.txt
└── LICENSE
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Saiful-Codes/cifar10-cnn-classifier.git
cd cifar10-cnn-classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Key Learnings

* Designing and implementing CNN architectures from scratch
* Structuring deep learning projects beyond notebooks
* Managing reproducible experiments
* Performing structured hyperparameter tuning
* Evaluating classification models beyond accuracy
* Converting research-style code into modular production-ready format
* Deploying PyTorch models using Streamlit


## Author

- Saiful Islam Shihab
- Bachelor of Computer Science (Artificial Intelligence Major)
- La Trobe University
