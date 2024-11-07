
from data_loader import fetch_data
from feature_engineering import calculate_features
from src.data_processing import split_data
from src.training import train_lstm_from_saved_data
from src.evaluation import evaluate_model
from tensorflow.keras.models import load_model
import numpy as np

ticker = "AAPL"
data = fetch_data(ticker=ticker)
data = calculate_features(data)
processed_data = data[['Close', 'EMA_20', 'EMA_50', 'RSI', '5_Day_Future_Volatility']]
split_data(processed_data)
model, model_run, X_test, y_test = train_lstm_from_saved_data()
best_model_path = "results/models/best_model.keras"
model.save(best_model_path)
best_model = load_model(best_model_path)
evaluate_model(best_model, X_test, y_test)
