import matplotlib.pyplot as plt

def plot_results(data, y_test, predictions):
    plt.figure(figsize=(14, 5))
    plt.plot(data.index[len(data) - len(y_test):], y_test, color='blue', label='Actual Stock Price')
    plt.plot(data.index[len(data) - len(y_test):], predictions, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

