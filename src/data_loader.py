import yfinance as yf
import os

def fetch_data(ticker="NVDA", start_date="2020-01-01", end_date="2023-01-01", save_path="../data/raw"):
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{ticker}_data.csv")
    data.to_csv(file_path)
    print(f"Data for {ticker} saved to {file_path}")
    return data
