# Volatility_prediction_with_deep_learning_methods

## Overview
This project focuses on predicting stock market volatility using LSTM neural networks. The model utilizes historical data and technical indicators to predict future volatility, which can assist in making informed investment and risk management decisions.

## Data Collection
- **Source**: Data is collected using `yfinance` Python library.
- **Date Range**: 2015-01-01 to 2023-01-01.
- **Features**: EMA, RSI, and rolling volatility.

## Model Architecture
The LSTM model is a stacked architecture:
- Two LSTM layers with 64 units each
- Dropout layers for regularization
- A dense output layer for predicting volatility

## Installation and Usage

### Requirements
- Python 3.x
- Required Libraries: See `requirements.txt`

### Running the Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nick7984/Volatility_prediction.git
