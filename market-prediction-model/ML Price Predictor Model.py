# Core data tools
import pandas as pd
import numpy as np

# Finance data
import yfinance as yf

# Technical analysis
import ta

# Preprocessing & scaling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Plotting
import matplotlib.pyplot as plt

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
import os
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Download 5 years of Apple stock data
df = yf.download("AAPL", start="2020-01-01", end="2025-01-01", auto_adjust=True)

# Flatten multi-index by taking just the first level of each tuple
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# Drop rows with any missing values
df.dropna(inplace=True)

# Add daily return
df['Return'] = df['Close'].pct_change()

# Add 10-day and 50-day moving averages
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

# Add 10-day rolling volatility
df['Volatility'] = df['Return'].rolling(window=10).std()

# Calculate True Range components
high_low = df['High'] - df['Low']
high_close_prev = (df['High'] - df['Close'].shift(1)).abs()
low_close_prev = (df['Low'] - df['Close'].shift(1)).abs()

# True Range
df['TR'] = high_low.combine(high_close_prev, max).combine(low_close_prev, max)

# ATR: Rolling mean of TR (usually 14 periods)
df['ATR'] = df['TR'].rolling(window=14).mean()

# Drop the True Range column as it's not needed after ATR calculation
df.drop(columns=['TR'], inplace=True)

# Add RSI (default 14 days)
df['RSI'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()

# Add MACD (12, 26, 9 by default)
macd = ta.trend.MACD(close=df['Close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()

# Add Bollinger Band width
bb = ta.volatility.BollingerBands(close=df['Close'])
df['BB_Width'] = bb.bollinger_wband()

# Add Lag features: past 20 days of Close prices
for i in range(1, 21):
    df[f'Lag{i}'] = df['Close'].shift(i)

# Drop NaNs
df.dropna(inplace=True)

# Define features and target
# Predict if price increases over next 10 days
future_return = df['Close'].shift(-10) / df['Close'] - 1
df['Target'] = (future_return > 0).astype(int)

# Drop the last row since it has no target
df.dropna(inplace=True)

# Define features (X) and target (y)
feature_cols=['Return', 'MA10', 'MA50', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'BB_Width', 'ATR'] + [f'Lag{i}' for i in range(1, 21)]
X = df[feature_cols].values
y = df['Target'].values

scaler=StandardScaler()
X_scaled= scaler.fit_transform(X)

# Function to create sequences for LSTM
def create_sequences(data, targets, seq_length=50):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x_seq = data[i:(i+seq_length)]
        y_seq = targets[i+seq_length]
        xs.append(x_seq)
        ys.append(y_seq)
    return np.array(xs), np.array(ys)

# X_scaled: standardized features (num_samples, num_features)
# y: labels (num_samples, )

seq_length = 50
X_seq, y_seq = create_sequences(X_scaled, y, seq_length)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(X_seq, y_seq):
    X_train_seq, X_test_seq = X_seq[train_idx], X_seq[test_idx]
    y_train_seq, y_test_seq = y_seq[train_idx], y_seq[test_idx]

# 2. Updated Model class
class LSTMStockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super(LSTMStockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        dropped = self.dropout(last_hidden)
        return self.fc(dropped)

# 3. Updated training loop with scheduler
input_dim = X_train_seq.shape[2]  # Number of features per timestep
model = LSTMStockPredictor(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # Reduce LR every 30 epochs

X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq.reshape(-1, 1), dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train loop
for epoch in range(100):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/100], Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    accuracy_metric = BinaryAccuracy()
    auc_metric = BinaryAUROC()
    acc = accuracy_metric(predictions, y_test_tensor)
    auc = auc_metric(predictions, y_test_tensor)
    print(f"Test Accuracy: {acc.item():.4f}")
    print(f"Test AUC: {auc.item():.4f}")

# Plotting
# Convert to NumPy arrays
# Convert to NumPy arrays
actual = y_test_tensor.numpy()
predicted = (predictions > 0.5).float().numpy()

# Select a subset window — last 100 points (or choose your own slice)
window_size = 100
start_idx = max(0, len(actual) - window_size)

actual_subset = actual[start_idx:]
predicted_subset = predicted[start_idx:]
x = np.arange(start_idx, start_idx + len(actual_subset))

plt.figure(figsize=(14, 6))

# Plot actual in blue
plt.stem(x, actual_subset, linefmt='b-', markerfmt='bo', basefmt=' ', label='Actual')

# For predicted, separate correct and incorrect
correct_idx = np.where(predicted_subset == actual_subset)[0]
incorrect_idx = np.where(predicted_subset != actual_subset)[0]

# Plot correct predictions in green
if len(correct_idx) > 0:
    plt.stem(x[correct_idx] + 0.2, predicted_subset[correct_idx], 
             linefmt='g-', markerfmt='go', basefmt=' ')

# Plot incorrect predictions in red
if len(incorrect_idx) > 0:
    plt.stem(x[incorrect_idx] + 0.2, predicted_subset[incorrect_idx], 
             linefmt='r-', markerfmt='ro', basefmt=' ')

plt.title(f"Apple Stock Price Direction Prediction (Last {window_size} Points)", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Direction (0 = Decrease, 1 = Increase)", fontsize=12)
plt.legend(['Actual', 'Predicted (Correct)', 'Predicted (Incorrect)'])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Save the model and prepare for deployment
torch.save(model.state_dict(), 'lstm_stock_predictor.pth')

# Load the model later for inference
model = LSTMStockPredictor(input_dim)
model.load_state_dict(torch.load('lstm_stock_predictor.pth'))
model.eval()

# Function to predict the next move for a given stock symbol
def predict_next_move(stock_symbol, model, scaler, feature_cols, seq_length=50):
    FLAT_THRESHOLD = 0.01            # 1% price change treated as flat
    CONFIDENCE_THRESHOLD = 0.6       # Minimum confidence to make a confident prediction

    # Step 1: Get recent data
    df = yf.download(stock_symbol, period='6mo', interval='1d', auto_adjust=True)

    # Step 2: Build same features
    df['Return'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Return'].rolling(window=10).std()

    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = np.maximum(np.maximum(tr1, tr2), tr3)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'].squeeze()).rsi()
    macd = ta.trend.MACD(close=df['Close'].squeeze())
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=df['Close'].squeeze())
    df['BB_Width'] = bb.bollinger_wband()

    for i in range(1, 21):
        df[f'Lag{i}'] = df['Close'].shift(i)

    # Step 3: Drop missing values
    df.dropna(inplace=True)

    # Step 4: Get last `seq_length` rows of features
    recent_data = df[feature_cols].tail(seq_length).values
    if recent_data.shape[0] < seq_length:
        raise ValueError("Not enough data to form a sequence")

    # Step 5: Scale features
    recent_scaled = scaler.transform(recent_data)

    # Step 6: Convert to tensor
    input_tensor = torch.tensor(recent_scaled.reshape(1, seq_length, -1), dtype=torch.float32)

    # Step 7: Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item()

    # Step 8: Estimate percentage price movement (naive estimation)
    price_now = df['Close'].iloc[-1]
    price_10_days_ago = df['Close'].iloc[-11] if len(df) >= 11 else df['Close'].iloc[0]
    percent_change = abs(price_now - price_10_days_ago) / price_10_days_ago

    # Step 9: Decide prediction with improved rules
    if prob < CONFIDENCE_THRESHOLD or percent_change < FLAT_THRESHOLD:
        prediction = 'Flat / Uncertain'
    elif prob >= 0.6:
        prediction = 'Increase'
    else:
        prediction = 'Decrease'

    print(f"{stock_symbol.upper()} → Prediction: {prediction} (Confidence: {prob:.4f}, Price Δ: {percent_change:.2%})")
    return prediction, prob

#Predict next move trial
#Enter your stock symbols here to predict
#Since the model is trained on Apple, it is best to use similar tech stocks
#Retrain with different stocks to use this model for other stocks
stocks = ["TSLA", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "NFLX", "ADSK", "ADBE", 'AMD', 'ORCL', 'IBM', 'QCOM', 'TXN', 'AVGO']

for stock in stocks:
    predict_next_move(stock, model, scaler, feature_cols)

