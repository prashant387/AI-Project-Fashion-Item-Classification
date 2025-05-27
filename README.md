# AI-Project-Fashion-Item-Classification
A Project using AI on Fashion Item Classification

# Fashion MNIST Classification using a Neural Network

This project demonstrates how to build and evaluate a neural network using TensorFlow/Keras to classify images from the Fashion MNIST dataset.

---

## 📌 Project Overview

Fashion MNIST is a dataset of 70,000 grayscale images of 10 fashion categories, each of size 28x28 pixels. This project implements a deep learning pipeline to classify the images into one of the following categories:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

---

## 🧠 Methodology

The project follows the standard machine learning pipeline:

1. **Dataset Loading**: 
   - The Fashion MNIST dataset is loaded using `tensorflow.keras.datasets.fashion_mnist`.

2. **Preprocessing**:
   - Pixel values are normalized to the range `[0, 1]`.
   - Labels are one-hot encoded for training with categorical cross-entropy loss.

3. **Model Architecture**:
   - A simple feedforward neural network (Multi-Layer Perceptron):
     - `Flatten`: Converts 28x28 image to 784-dimensional vector.
     - `Dense(128, relu)`: Fully connected layer with 128 neurons.
     - `Dense(64, relu)`: Fully connected layer with 64 neurons.
     - `Dense(10, softmax)`: Output layer with 10 classes.

4. **Training**:
   - Optimizer: `Adam`
   - Loss Function: `categorical_crossentropy`
   - Epochs: 5
   - Validation Split: 10%

5. **Evaluation**:
   - Test accuracy
   - Classification report (precision, recall, F1-score)
   - Confusion matrix

6. **Visualization**:
   - Bar chart for per-class recall
   - Confusion matrix heatmap
   - Optional: Sample prediction visualizations

---

## 🚀 How to Run

### Requirements

- Python 3.x
- TensorFlow
- NumPy
- pandas
- seaborn
- matplotlib
- scikit-learn

📊 Results
Overall Test Accuracy: ~87–89% (may vary slightly)

Per-Class Accuracy: Visualized using bar charts and classification reports.

Confusion Matrix: Displays how predictions are distributed across classes.

📷 Sample Outputs
Per-Class Accuracy Bar Chart


Confusion Matrix


📚 Future Improvements
Implement CNNs (Convolutional Neural Networks) for better performance.

Add early stopping and learning rate scheduling.

Use data augmentation to improve generalization.

Deploy the model using Flask or Streamlit.

🤝 Acknowledgements
Dataset from Fashion MNIST

TensorFlow/Keras for model development
