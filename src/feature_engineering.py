import pandas as pd

def calculate_features(data):

    data["Return"] = data["Close"].pct_change()
    data["5_Day_Future_Volatility"] = data["Return"].rolling(window=5).std().shift(-5) #volatilityy
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()

    delta = data['Close'].diff() # RSI
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    data = data.dropna()
    return data
