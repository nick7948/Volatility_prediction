import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error
import os

def evaluate_model(model, X_test, y_test, dir="results/figures"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False) 
    print("Test MSE:", mse)

    os.makedirs(dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Volatility", color="blue")
    plt.plot(y_pred, label="Predicted Volatility", color="red")
    plt.xlabel("Time Step")
    plt.ylabel("7-Day Future Volatility")
    plt.title("LSTM Predictions vs. Actual")
    plt.legend()
    plt.savefig(os.path.join(dir, "predictions_vs_actual.png"))
    plt.show()
