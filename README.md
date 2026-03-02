# MNIST Digit Identification

[Streamlit Cloud](https://mnist-number-identification.streamlit.app/) | [Repository](https://github.com/ghanap/MNIST-number-identification)

## Overview
This project is a web-based application designed to classify handwritten digits (0-9) using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset, a benchmark in the field of computer vision. The application provides an intuitive interface for users to upload images or draw digits and receive real-time classification results.

---

## Technical Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow, Keras |
| **Web Framework** | Streamlit |
| **Image Processing** | OpenCV, NumPy, Pillow |
| **Data Analysis** | Matplotlib |

---

## Key Features
* **High Accuracy:** Utilizes a CNN architecture optimized for handwritten digit recognition with near 99% accuracy.
* **Real-Time Inference:** Fast processing of input data for immediate classification.
* **Confidence Visualization:** Displays the probability distribution for all classes, showing the model's confidence levels for each digit.
* **Flexible Input:** Supports image file uploads and (if configured) digital canvas input for direct interaction.

---

## Model Architecture
The underlying model follows a sequential CNN structure:
1.  **Input Layer:** 28x28 grayscale images.
2.  **Convolutional Layers:** To extract spatial features and patterns.
3.  **Pooling Layers:** To reduce dimensionality and computational load.
4.  **Dense Layers:** Fully connected layers for high-level reasoning.
5.  **Output Layer:** Softmax activation for 10-class probability distribution.

---

## Model Strategies: Old vs New

### Previous Strategy (Old Version)
The initial version of the project used a combination of Principal Component Analysis (PCA) and a Scaler for preprocessing the input images before classification. This approach aimed to reduce dimensionality and normalize pixel values, which helped traditional machine learning models but limited the achievable accuracy for complex image data like MNIST digits.

- **Preprocessing:** PCA for dimensionality reduction, Scaler for normalization
- **Model:** Simpler classifier (e.g., SVM, shallow neural net)
- **Accuracy:** Approximately 93%
- **Limitations:** Lost some spatial information, less effective for handwritten digit nuances

### Improved Strategy (Current Version)
The new version leverages a deep Convolutional Neural Network (CNN) without PCA or Scaler preprocessing. The raw 28x28 grayscale images are directly fed into the CNN, which automatically learns spatial features and patterns crucial for digit recognition.

- **Preprocessing:** Direct grayscale conversion and normalization (no PCA/Scaler)
- **Model:** Deep CNN (see Model Architecture above)
- **Accuracy:** Near 99% on MNIST
- **Advantages:** Preserves spatial information, better generalization, higher accuracy

---

## Accuracy Improvements
The transition from the old strategy (PCA + Scaler + simple classifier, ~93% accuracy) to the new strategy (deep CNN, ~99% accuracy) resulted in a significant boost in accuracy and robustness. The CNN model is able to capture complex patterns in handwritten digits, outperforming traditional approaches.

---

## Project Structure

The repository is organized as follows to ensure modularity and ease of deployment:

