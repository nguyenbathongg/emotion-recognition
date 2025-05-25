# My Emotion Recognition Project


**CNN model for classifying facial emotions into seven categories using Keras**

A deep learning model for facial emotion classification. This Keras-based project includes a CNN architecture and pre-trained weights for quick testing on custom images.

---

<!-- TABLE OF CONTENTS -->
<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#model-architecture">Model Architecture</a></li>
    <li><a href="#implementation-details">Implementation Details</a></li>
    <li><a href="#training-process">Training Process</a></li>
    <li><a href="#performance-metrics">Performance Metrics</a></li>
    <li><a href="#run">Run</a></li>
    <li><a href="#data">Data</a></li>
    <li><a href="#future-improvements">Future Improvements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is the final project created for the Spring semester of 2023 course in Introduction to Artificial Intelligence with a focus on the study of Machine Learning.

The project implements a Convolutional Neural Network (CNN) model for facial emotion classification. The system analyzes facial expressions from images and classifies them into one of seven emotional categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

During the project development iterative process, many different machine learning ideas were tested and evaluated based on the performance metrics. The implementation uses Keras with TensorFlow backend and achieves competitive accuracy compared to human performance in emotion recognition tasks.


## Model Architecture

The final model architecture consists of multiple convolutional layers followed by fully connected layers. The architecture was iteratively refined to improve performance and reduce overfitting.

### Network Architecture Details

| Layer | Layer Type        | Kernel Size (KxK) | Stride (SxS) | Filters | Output Dimension | Activation |
|-------|-------------------|-------------------|--------------|---------|------------------|------------|
| 1     | Conv2D            | 3×3               | 1×1          | 32      | 48×48×32         | ReLU       |
|       | BatchNormalization| -                 | -            | -       | -                | -          |
| 2     | Conv2D            | 3×3               | 1×1          | 64      | 48×48×64         | ReLU       |
|       | BatchNormalization| -                 | -            | -       | -                | -          |
| 3     | MaxPooling2D      | 2×2               | 2×2          | -       | 24×24×64         | -          |
| 4     | Conv2D            | 3×3               | 1×1          | 128     | 24×24×128        | ReLU       |
|       | BatchNormalization| -                 | -            | -       | -                | -          |
| 5     | MaxPooling2D      | 2×2               | 2×2          | -       | 12×12×128        | -          |
| 6     | Conv2D            | 3×3               | 1×1          | 256     | 12×12×256        | ReLU       |
|       | BatchNormalization| -                 | -            | -       | -                | -          |
| 7     | MaxPooling2D      | 2×2               | 2×2          | -       | 6×6×256          | -          |
| 8     | Flatten           | -                 | -            | -       | 9216             | -          |
| 9     | Dense             | -                 | -            | -       | 256              | ReLU       |
|       | BatchNormalization| -                 | -            | -       | -                | -          |
|       | Dropout (0.25)    |                   |              |         |                  |            |
| 10    | Dense             | -                 | -            | -       | 128              | ReLU       |
|       | BatchNormalization| -                 | -            | -       | -                | -          |
|       | Dropout (0.25)    |                   |              |         |                  |            |
| 11    | Dense             | -                 | -            | -       | 7                | Softmax    |

The model uses several techniques to improve performance and generalization:

- **Batch Normalization**: Applied after convolutional and dense layers to stabilize and accelerate training
- **Dropout**: Used in fully connected layers to prevent overfitting
- **Data Augmentation**: Horizontal and vertical flipping to increase training data diversity

## Implementation Details

### Dependencies

The project relies on the following key libraries:

- TensorFlow/Keras: Framework for building and training the neural network
- NumPy: For array operations and data manipulation
- Matplotlib: For visualization of training progress and results
- PIL (Python Imaging Library): For image processing tasks

### Hyperparameters

The final model uses the following hyperparameters:

- **Batch size**: 32
- **Epochs**: 10
- **Optimizer**: Adam
  - Learning rate: 0.001
  - Beta1: 0.9
  - Beta2: 0.999
  - Epsilon: 1e-07
- **Loss function**: Categorical cross-entropy
- **Metrics**: Accuracy

## Training Process

### Data Preprocessing

- **Normalization**: Pixel values are normalized from the range [0-255] to [0-1]
- **Data Augmentation**: The training data generator applies horizontal and vertical flipping to increase the diversity of the training set
- **Image Resizing**: All images are resized to 48×48 pixels

### Training Strategy

The model was trained with an iterative approach, with each iteration refining the architecture and hyperparameters based on performance. Key strategies implemented:

1. **Checkpoint System**: Implemented to save the model weights that result in the highest validation accuracy
2. **Regularization Techniques**: Batch normalization and dropout were used to prevent overfitting
3. **Architectural Experimentation**: Multiple model architectures were tested to find the optimal balance between complexity and performance

## Performance Metrics

The final model achieved the following performance metrics:

- **Training accuracy**: ~70-75%
- **Validation accuracy**: ~65-70%

The model's performance approaches human-level performance in similar tasks, which is typically around 82% according to research.

### Challenges and Solutions

- **Overfitting**: Initially, the model showed signs of overfitting with high training accuracy but lower validation accuracy. This was addressed by implementing dropout and batch normalization.
- **Early Convergence**: The validation loss was found to converge slightly faster than the training loss and barely improved after epoch 10. Various regularization techniques were tested to address this issue.

<!-- RUN -->
## Run

Open the notebook using your choice software in a terminal or command window by navigating to the top-level project directory, `emotion-recognition`. For example, if the software is Jupyter Notebook:

```bash
jupyter notebook emotion_recognition.ipynb
```

The project folder includes the weights of the trained neural network and the model can be tested on custom images without requiring to train it again.



<!-- DATA -->
## Data

This dataset is a modified version of the Emotion Detection dataset found on [Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer), consisting of 35887 data entries, classified into 7 categories.

The dataset includes:

- **Total images**: 35,887 facial expression images
- **Image dimensions**: 48×48 pixels
- **Color format**: Grayscale
- **Classes**: 7 emotion categories
- **Distribution**: 
  - Training set: ~80% (28,709 images)
  - Validation set: ~10% (3,587 images)
  - Test set: ~10% (3,591 images)



## Future Improvements

Potential enhancements for future versions of the project:

1. **Transfer Learning**: Experiment with pre-trained models (e.g., VGG, ResNet) as feature extractors
2. **Architecture Refinement**: Further optimize the CNN architecture, potentially with techniques like residual connections
3. **Data Expansion**: Incorporate additional datasets to improve generalization
4. **Model Compression**: Explore techniques to reduce model size for deployment on edge devices
5. **Real-time Implementation**: Extend the system to work with real-time video feeds for dynamic emotion recognition
6. **Cross-cultural Validation**: Test and refine the model across diverse demographic groups to ensure fair performance
7. **Ensemble Methods**: Combine multiple models to improve overall accuracy and robustness






