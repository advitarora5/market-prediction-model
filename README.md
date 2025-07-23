# market-prediction-model
**Project Overview**

This project implements a machine learning model that predicts whether a stock’s price will increase or decrease over the next 10 trading days. The model is based on a Long Short-Term Memory (LSTM) neural network architecture, which is well-suited for analyzing sequential data such as stock prices.

The model is trained using 5 years of historical daily price data for Apple Inc. (AAPL) and tested on 15 sample technology stocks that have similar market capitalization to Apple. This allows us to evaluate the model’s generalizability to stocks with comparable characteristics.

**Key Performance Metrics**

Model's Average Accuracy: 0.849 (84.9%)
This means that the model correctly predicts the direction of stock price movement (up or down) approximately 85% of the time on the test set. Accuracy measures the proportion of correct predictions among all predictions made.

Model's Average Area Under the Curve (AUC): 0.9187
The AUC refers to the area under the Receiver Operating Characteristic (ROC) curve. It measures the model’s ability to distinguish between the two classes (price increase vs. decrease) across different classification thresholds. A value of 0.9187 indicates excellent discriminative power, meaning the model is very effective at ranking positive cases higher than negative cases.

**Understanding Loss Per Epoch**

During training, the model optimizes a loss function (typically binary cross-entropy for classification tasks). The loss per epoch represents the average error between the model's predicted outputs and the actual labels for one full pass over the training data.

A decreasing loss over epochs indicates that the model is learning and improving its predictions.

If the loss plateaus or increases, it may suggest overfitting or that the model is no longer improving.

**How It Works**

Input Data: The model uses sequential historical price data, including features such as daily closing prices, volume, and other relevant technical indicators derived from AAPL's 5-year dataset.

Model Architecture: An LSTM network captures temporal dependencies and patterns in the price data to predict future trends.

Prediction Output: For each input sequence, the model outputs a probability indicating whether the stock price will increase or decrease over the next 10 trading days.

**Test Results**

The model was tested on 15 technology stocks chosen based on similar market capitalization to Apple. The results demonstrate strong predictive performance, suggesting the model’s potential for application to other large-cap tech stocks.

**Usage**

Clone the repository:
    git clone https://github.com/advitarora5/market-prediction-model.git

Install required Python packages:
    
    pip install -r requirements.txt

Prepare the dataset (see data/ folder and instructions).

Run the training script:
    
    python train_model.py

Evaluate the model on test data:
    
    python evaluate_model.py