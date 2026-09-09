# Credit Card Fraud Detection using Artificial Neural Network

A Credit Card Fraud Detection system built using an Artificial Neural Network (ANN) with TensorFlow/Keras. The project focuses on detecting fraudulent transactions in a highly imbalanced dataset and evaluating the model using fraud-focused performance metrics.

## Project Overview

Credit card fraud detection is a binary classification problem where the number of legitimate transactions is significantly higher than fraudulent transactions.

This project uses an Artificial Neural Network to classify transactions as either:

* Normal
* Fraudulent

Because of the severe class imbalance, accuracy alone is not considered a reliable evaluation metric. The project therefore focuses on Precision, Recall, F1-Score, AUPRC, and the Confusion Matrix.

## Dataset

The project uses the Credit Card Fraud Detection dataset containing transactions made by European cardholders.

Dataset characteristics:

* 284,807 transactions
* 492 fraudulent transactions
* 30 input features
* `Class` as the target variable
* Approximately 0.17% of transactions are fraudulent

The `V1`–`V28` features are anonymized numerical features provided by the dataset.

Dataset source:

[Credit Card Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?utm_source=chatgpt.com)


## Technologies Used

* Python
* TensorFlow
* Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Streamlit
* Joblib

## Machine Learning Workflow

The project follows the following workflow:

```text
Data Loading
    ↓
Exploratory Data Analysis
    ↓
Train/Test Split
    ↓
Data Preprocessing
    ↓
Class Weight Calculation
    ↓
Artificial Neural Network
    ↓
Model Training
    ↓
Prediction
    ↓
Precision-Recall Analysis
    ↓
Threshold Optimization
    ↓
Final Evaluation
```

## Exploratory Data Analysis

The dataset was analyzed to understand:

* Dataset dimensions
* Missing values
* Feature information
* Class distribution
* Transaction amount distribution
* Amount distribution across normal and fraudulent transactions

## Data Preprocessing

The dataset is divided into training and testing sets using a stratified split to preserve the original class distribution.

`Time` and `Amount` are standardized using `StandardScaler`.

The scaler is fitted only on the training data and then applied to the test data to prevent data leakage.

The `V1`–`V28` features are already transformed/anonymized numerical features and are not scaled again.

## Handling Class Imbalance

The dataset contains a very small number of fraudulent transactions compared to normal transactions.

Class weights are calculated using Scikit-learn's `compute_class_weight` and passed to the ANN during training.

This gives greater importance to the minority fraud class during model optimization.

## Artificial Neural Network

The model is implemented using TensorFlow/Keras.

Architecture:

```text
Input Layer
    ↓
Dense Layer: 64 neurons, ReLU
    ↓
Dropout: 30%
    ↓
Dense Layer: 32 neurons, ReLU
    ↓
Dropout: 30%
    ↓
Dense Layer: 16 neurons, ReLU
    ↓
Dropout: 30%
    ↓
Output Layer: 1 neuron, Sigmoid
```

Training configuration:

* Optimizer: Adam
* Loss Function: Binary Cross-Entropy
* Batch Size: 2048
* Epochs: 100
* Class Weighting: Enabled
* Dropout: 30%

## Model Evaluation

The model is evaluated using metrics that are more appropriate for highly imbalanced classification problems.

### Precision

Measures how many transactions predicted as fraud were actually fraudulent.

### Recall

Measures how many actual fraudulent transactions were successfully detected.

### F1-Score

Provides a balance between Precision and Recall.

### AUPRC

The Area Under the Precision-Recall Curve provides an overall view of the model's Precision-Recall performance across different classification thresholds.

### Confusion Matrix

The confusion matrix provides:

* True Negatives
* False Positives
* False Negatives
* True Positives

## Classification Threshold Optimization

The ANN produces a probability between 0 and 1 rather than directly producing a final class.

Instead of relying only on the default threshold of 0.5, multiple thresholds are evaluated using the Precision-Recall curve.

The threshold that produces the highest F1-score is selected as the best threshold for the final prediction.

```text
ANN Probability
       ↓
Different Thresholds
       ↓
Precision + Recall
       ↓
F1-Score
       ↓
Best Threshold
       ↓
Normal / Fraud
```

## Streamlit Application

A Streamlit interface is included to demonstrate the trained ANN model.

The interface provides a simplified transaction form using:

* Transaction Amount
* Transaction Time

The trained ANN model, fitted scaler, and optimized classification threshold are loaded from saved files.

The application returns:

* Fraud probability
* Classification result
* Decision threshold
* Model prediction details

The Streamlit interface is intended as an educational demonstration rather than a production financial fraud detection system.

## Project Structure

```text
CreditCard-Predictor/
│
├── app.py
├── train_model.ipynb
├── model.keras
├── scaler.pkl
├── threshold.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CreditCard-Predictor
```

Activate the required environment:

```bash
conda activate conda-dl
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

## Model Files

The project uses the following saved files:

* `model.keras` — trained Artificial Neural Network
* `scaler.pkl` — fitted StandardScaler
* `threshold.pkl` — optimized classification threshold

These files are generated after training the model.

## Dataset Availability

The original dataset is not included in this repository because of its size and licensing/distribution considerations.

The dataset should be obtained separately and placed in the project directory before running the training notebook.

## Limitations

This project is primarily an educational and demonstration project.

The `V1`–`V28` features in the dataset are anonymized, which makes them unsuitable for a conventional human-friendly transaction form.

The Streamlit interface therefore provides a simplified demonstration of the trained model rather than a complete production fraud detection system.

The model should not be used for real financial decisions without further validation, feature engineering, extensive testing, monitoring, and domain-specific evaluation.

## Learning Objectives

This project was developed to practice and understand:

* Artificial Neural Networks
* Binary Classification
* Imbalanced Classification
* Class Weights
* Dropout Regularization
* TensorFlow/Keras
* Model Evaluation
* Precision
* Recall
* F1-Score
* Precision-Recall Curves
* AUPRC
* Classification Threshold Optimization
* Confusion Matrix
* Streamlit Deployment

## Author

Burhan Arshad

BS Computer Science Student
