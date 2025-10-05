# ✍️ Handwritten Digit Recognition (CNN)

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) using the industry-standard **MNIST dataset**.

## 🚀 Key Features
* **Deep Learning Model:** Utilizes a custom-built CNN for high-accuracy image classification.
* **Data Preprocessing:** Includes data normalization and reshaping for optimized model input.
* **Iterative Training:** The model was trained and tuned across multiple training runs to observe the impact of key hyperparameters (epochs) on performance.

## 🛠 My Contribution & Development Log

My primary work involved setting up the training pipeline, customizing the model architecture, and conducting iterative training to optimize the final model accuracy.

| Commit Detail | Epochs | Output File | Rationale |
| :--- | :--- | :--- | :--- |
| **Initial Base** | 3 | `mnist_model_e3.h5` | Baseline training to ensure the CNN and data pipeline were functioning correctly. |
| **First Tune** | 5 | `mnist_model_e5.h5` | Increased epochs to reduce underfitting; observed a notable jump in validation accuracy. |
| **Final Training** | 8 | `mnist_model_e8.h5` | Finalized training with 8 epochs to balance high accuracy with preventing overfitting on the training data. |

## ⚙️ Setup and Dependencies

To run this project locally (if needed), ensure you have Python 3 and the following libraries installed:

```bash
pip install tensorflow numpy matplotlib