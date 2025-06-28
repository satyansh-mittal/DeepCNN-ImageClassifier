# Deep CNN Image Classifier

This project demonstrates how to build, train, and evaluate a deep convolutional neural network (CNN) for image classification using TensorFlow and Keras. The example provided in the notebook classifies images into 'happy' and 'sad' categories, but the framework can be adapted for any binary image classification task.

## Table of Contents
- [Project Structure](#project-structure)
- [Features](#features)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Workflow](#workflow)
- [Model Architecture](#model-architecture)
- [Results](#results)

## Project Structure

```
.
├── data/
│   ├── happy/
│   │   ├── image1.jpg
│   │   └── ...
│   └── sad/
│       ├── image2.jpg
│       └── ...
├── model/
│   └── happysadmodel.h5
├── logs/
├── ImageClassificationCNN.ipynb
└── README.md
```

- `data/`: Contains the image dataset, organized into subdirectories for each class.
- `model/`: Stores the trained Keras model.
- `logs/`: Directory for TensorBoard logs generated during training.
- `ImageClassificationCNN.ipynb`: The main Jupyter Notebook with all the code.

## Features

- **Data Cleaning**: Automatically removes corrupted images or files with unsupported extensions.
- **Data Loading & Preprocessing**: Efficiently loads and preprocesses image data using `tf.data.Dataset`.
- **CNN Model**: A custom-built CNN model using Keras Sequential API.
- **Training & Evaluation**: Trains the model and evaluates its performance using precision, recall, and accuracy.
- **Inference**: Demonstrates how to use the trained model to predict the class of a new image.
- **Model Saving**: Saves the trained model for future use.

## Dependencies

The project is built using Python and the following libraries. You can install them using pip:

```bash
pip install tensorflow opencv-python matplotlib
```

- TensorFlow
- OpenCV (`opencv-python`)
- Matplotlib
- NumPy (usually installed with TensorFlow)

## Usage

1.  **Clone the repository.**
2.  **Prepare your dataset**: Create a `data` directory and inside it, create subdirectories for each of your classes (e.g., `happy`, `sad`). Place the respective images in these folders.
3.  **Run the Jupyter Notebook**: Open and run the cells in [`ImageClassificationCNN.ipynb`](c:\Users\mukes\OneDrive\Desktop\NOTES\AIML Course\Projects\Deep CNN Image Classifier\ImageClassificationCNN.ipynb) to train the model on your data.

## Workflow

The notebook follows these steps:

1.  **Setup**: Imports necessary libraries and configures the GPU.
2.  **Data Validation**: Scans the `data` directory, checks image integrity, and removes invalid files.
3.  **Load Data**: Loads images into a `tf.data.Dataset` using `tf.keras.utils.image_dataset_from_directory`.
4.  **Preprocess Data**:
    -   Scales pixel values to a range of [0, 1].
    -   Splits the data into training (70%), validation (20%), and test (10%) sets.
5.  **Build Model**: Defines the CNN architecture.
6.  **Train Model**: Compiles and trains the model using the training and validation sets. Training progress is logged for TensorBoard.
7.  **Evaluate Performance**: Plots accuracy/loss curves and calculates precision, recall, and accuracy on the test set.
8.  **Test on a New Image**: Demonstrates how to load a single image, preprocess it, and get a prediction from the model.
9.  **Save Model**: Saves the final trained model to the `model` directory.

## Model Architecture

The model is a Sequential CNN with the following layers:

| Layer             | Output Shape        | Param # |
|-------------------|---------------------|---------|
| Conv2D            | (None, 254, 254, 16) | 448     |
| MaxPooling2D      | (None, 127, 127, 16) | 0       |
| Conv2D            | (None, 125, 125, 32) | 4640    |
| MaxPooling2D      | (None, 62, 62, 32)  | 0       |
| Conv2D            | (None, 60, 60, 16)  | 4624    |
| MaxPooling2D      | (None, 30, 30, 16)  | 0       |
| Flatten           | (None, 14400)       | 0       |
| Dense             | (None, 256)         | 3686656 |
| Dense (Sigmoid)   | (None, 1)           | 257     |

**Total params**: 3,696,625

## Results

The model is trained for 20 epochs. The training and validation accuracy/loss are plotted to monitor performance and check for overfitting. The final model is evaluated on the test set to report its precision, recall, and accuracy.
