from src.data_preprocessing import fetch_data, preprocess_data
from src.model import build_model
from src.train import train_model
from src.predict import make_predictions
from src.visualization import plot_results

def main():
    # Step 1: Fetch data
    data = fetch_data('AAPL', '2015-01-01', '2023-01-01')
    
    # Step 2: Preprocess data
    X, y, scaler = preprocess_data(data)
    
    # Step 3: Build model
    model = build_model(input_shape=(X.shape[1], 1))
    
    # Step 4: Train model
    train_model(model, X, y)
    
    # Step 5: Make predictions
    predictions = make_predictions(model, X, scaler)
    
    # Step 6: Plot results
    plot_results(data, y, predictions)

if __name__ == '__main__':
    main()

