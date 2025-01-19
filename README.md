# Animal Classification Project

## Project Overview

This project shows deep learning to classify animals in images. The model is trained using a dataset of labeled images and is implemented using TensorFlow/Keras. The dataset is split into training, validation, and test sets for training, model evaluation, and testing the performance of the trained model.

## Table of Contents
1. [Project Overview](#project-overview)
4. [Dataset](#dataset)
5. [Training the Model](#training-the-model)
6. [Evaluating the Model](#evaluating-the-model)
7. [File Structure](#file-structure)
8. [License](#license)

## Dependencies

To run this project, ensure you have Python installed along with the following libraries:

- **TensorFlow** (for training and evaluating the model)
- **Keras** (for model building)
- **scikit-learn** (for dataset splitting)


Dataset
-------

The dataset used in this project contains images of various animals. The images are split into the following categories:

*   **Training Data**: 80% of the total dataset used for training the model.
    
*   **Validation Data**: 10% used for model validation during training.
    
*   **Test Data**: 10% used for evaluating the final model.
    

The prepared dataset structure should look like this:

```bash
processed_dataset/
├── train/
├── validation/
└── test/
```
Training the Model
----------------
To train the model, run the following command:

```bash
python main.py
```

### Hyperparameters:

*   **Epochs**: Set to 10 (You can modify this value in the code).
    
*   **Batch Size**: 32 (Also adjustable).
    
*   **Model**: A CNN built with multiple convolutional layers, max pooling, and dense layers.
    

### The process will:

*   Load and preprocess the data.
    
*   Build the CNN model.
    
*   Train the model and save the best model using checkpoints.
    

You can adjust training parameters such as epochs and batch size in the code.

Evaluating the Model
--------------------

After training the model, evaluate it using the following command:

```bash
python evaluate_model.py
```
This will:

*   Load the best-trained model.
    
*   Evaluate its performance on the test data.
    
*   Print the accuracy and loss of the model on the test dataset.

License
-------

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ragul-rofi/AnimalClassificationML/blob/main/LICENSE) file for details.

### Notes:

*   **Git LFS**: For large files like the .keras model file, use [Git Large File Storage (LFS)](https://git-lfs.github.com/). Follow the instructions in the Git LFS documentation to set it up.
    
*   **Data Privacy**: Make sure you have the appropriate rights and permissions to share or use the dataset if it contains sensitive information.
    
*   **Dataset Download**: If you need to download an existing dataset, refer to the dataset page or use any other public animal classification dataset.

