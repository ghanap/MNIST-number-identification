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

## Project Structure

The repository is organized as follows to ensure modularity and ease of deployment:

```text
MNIST-number-identification/
├── app.py                # Main entry point for the Streamlit web application
├── models/               # Contains serialized model files (.h5, .keras, or .pkl)
├── notebooks/            # Jupyter notebooks for data exploration and model training
├── src/                  # (Optional) Source scripts for data preprocessing and utility functions
├── requirements.txt      # Comprehensive list of Python dependencies
└── README.md             # Project documentation and setup guide
git clone [https://github.com/ghanap/MNIST-number-identification.git](https://github.com/ghanap/MNIST-number-identification.git)
cd MNIST-number-identification
