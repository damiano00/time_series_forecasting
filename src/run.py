import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_processor import DataProcessor
from src.lstm.lstm import LSTM


# CUDA availability
def check_cuda(use_gpu=False):
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        if use_gpu:
            print("CUDA Version:", torch.version.cuda)
            print("GPU Name:", torch.cuda.get_device_name(0))
            torch.device("cuda")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print("Selected primary: GPU")
        else:
            torch.device("cpu")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("Selected primary: CPU")


def train_and_validate(model, criterion, optimizer, train_loader, test_loader, num_epochs, result_dir,
                       specific_stock_idx, stock_test_name):
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        try:
            # Training Phase
            model.train()
            for X_train_batch, y_train_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_train_batch)
                loss = criterion(outputs, y_train_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation Phase
            model.eval()
            with torch.no_grad():
                for X_test_batch, y_test_batch in test_loader:
                    val_outputs = model(X_test_batch)
                    val_loss = criterion(val_outputs, y_test_batch)
                    test_losses.append(val_loss.item())

            # Real-time loss visualization
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{result_dir}/loss_plot.png")
            plt.close()

            # Save model checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f"{result_dir}/model_checkpoint_epoch_{epoch + 1}.pth")

            # Plot prediction vs actual for a specific stock
            plt.figure(figsize=(10, 5))
            plt.plot(y_test_batch[:, specific_stock_idx].cpu().numpy(), label='Actual')
            plt.plot(val_outputs[:, specific_stock_idx].cpu().numpy(), label='Predicted')
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.title('Prediction vs Actual')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{result_dir}/prediction_plot.png")
            plt.close()

            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving model...")
            torch.save(model.state_dict(), f"{result_dir}/model_interrupted.pth")
            break

    # Calculate and save metrics after training
    y_test_np = y_test_batch[:, specific_stock_idx].cpu().numpy()
    val_outputs_np = val_outputs[:, specific_stock_idx].cpu().numpy()

    mae = mean_absolute_error(y_test_np, val_outputs_np)
    mse = mean_squared_error(y_test_np, val_outputs_np)
    r2 = r2_score(y_test_np, val_outputs_np)

    metrics_dir = f"{result_dir}/{stock_test_name}"
    os.makedirs(metrics_dir, exist_ok=True)
    with open(f"{metrics_dir}/metrics.txt", "w") as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"R2 Score: {r2}\n")

    # Save the prediction vs actual plot in the stock-specific folder
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_np, label='Actual')
    plt.plot(val_outputs_np, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.title(f'Close Price Prediction vs Actual for {stock_test_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{metrics_dir}/close_price_plot.png")
    plt.close()


def main(data_path, num_csvs, cols_to_scale, cols_to_normalize, stock_test_name='AAPL.csv', num_epochs=2000, seq_len=60,
         batch_size=32,  use_gpu=True):
    # Directories for saving results
    results_path = f"../src/lstm/test_result_{num_csvs}"
    os.makedirs(results_path, exist_ok=True)

    # Data Processor
    dp = DataProcessor(
        dir_path=data_path,
        seq_len=seq_len,
        test_size=0.1,
        cols_to_scale=cols_to_scale,
        cols_to_normalize=cols_to_normalize,
        batch_size=batch_size
    )

    # Test csvs = 50
    names_50 = ['aal.csv', 'AAPL.csv', 'ABBV.csv', 'AMD.csv', 'amgn.csv', 'AMZN.csv', 'BABA.csv',
                'bhp.csv', 'bidu.csv', 'biib.csv', 'BRK-B.csv', 'C.csv', 'cat.csv', 'cmcsa.csv', 'cmg.csv',
                'cop.csv', 'COST.csv', 'crm.csv', 'CVX.csv', 'dal.csv', 'DIS.csv', 'ebay.csv', 'GE.csv',
                'gild.csv', 'gld.csv', 'GOOG.csv', 'gsk.csv', 'INTC.csv', 'KO.csv', 'mrk.csv', 'MSFT.csv',
                'mu.csv', 'nke.csv', 'nvda.csv', 'orcl.csv', 'pep.csv', 'pypl.csv', 'qcom.csv', 'QQQ.csv',
                'SBUX.csv', 'T.csv', 'tgt.csv', 'tm.csv', 'TSLA.csv', 'TSM.csv', 'uso.csv', 'v.csv', 'WFC.csv',
                'WMT.csv', 'xlf.csv']

    # Test csvs = 25
    names_25 = ['AAPL.csv', 'ABBV.csv', 'AMZN.csv', 'BABA.csv', 'BRK-B.csv', 'C.csv', 'COST.csv', 'CVX.csv',
                'DIS.csv', 'GE.csv', 'INTC.csv', 'MSFT.csv', 'nvda.csv', 'pypl.csv', 'QQQ.csv', 'SBUX.csv', 'T.csv',
                'TSLA.csv', 'WFC.csv', 'KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']

    # Test csvs = 5
    names_5 = ['KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']

    check_cuda(use_gpu=use_gpu)

    # Build dataframe
    if num_csvs == 5 or num_csvs == 25 or num_csvs == 50:
        dp.csv_to_df(stocks_list=names_5 if num_csvs == 5 else names_25 if num_csvs == 25 else names_50)
    else:
        raise ValueError("The number of CSVs is invalid, please choose from 5, 25, or 50.")

    # Split the data
    dp.split_df()

    # Scale and normalize the train and test data
    dp.feature_transformation(dp.train_data)
    dp.feature_transformation(dp.test_data, False)

    # Get the train and test DataLoader objects
    train_loader = dp.get_train_loader()
    test_loader = dp.get_test_loader()

    # Model parameters
    input_size = next(iter(train_loader))[0].shape[2]  # Dynamically get input size from DataLoader
    hidden_size = 64
    num_layers = 2
    output_size = next(iter(train_loader))[1].shape[1]  # Number of labels (Close and Adj Close)
    learning_rate = 0.001


    # Model, Loss, Optimizer
    model = LSTM(input_size, hidden_size, num_layers, output_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation
    train_and_validate(model, criterion, optimizer, train_loader, test_loader, num_epochs, results_path,
                       specific_stock_idx=0, stock_test_name=stock_test_name)


if __name__ == "__main__":
    main("../dataset/processed/data_for_lstm",
         5,
         [1, 2, 3, 4, 5, 6],
         [1, 2, 3, 4, 5, 6],
         num_epochs=2000,
         stock_test_name='GOOG.csv',
         seq_len=60,
         batch_size=32,
         use_gpu=True)
