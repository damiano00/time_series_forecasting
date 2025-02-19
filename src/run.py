import os
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_processor import DataProcessor
from src.lstm.lstm import LSTM
from src.utils import Environment


def save_model_checkpoint(model, epoch, result_dir):
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f"{result_dir}/model_checkpoint_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)


def calculate_metrics(test_loader, model):
    y_true = torch.cat([y for _, y in test_loader], dim=0)
    y_pred = torch.cat([model(X).detach() for X, _ in test_loader], dim=0)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mae, mse, r2


def save_metrics(metrics_dir, mae, mse, r2):
    os.makedirs(metrics_dir, exist_ok=True)
    with open(f"{metrics_dir}/metrics.txt", "w") as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"R2 Score: {r2}\n")


def train_and_validate(model, criterion, optimizer, train_loader, test_loader, num_epochs, result_dir, test_stock_name):
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        try:
            # Training Phase
            model.train()
            for X_train_batch, y_train_batch in train_loader:
                loss = train_step(model, criterion, optimizer, X_train_batch, y_train_batch)
                train_losses.append(loss)

            # Validation Phase
            val_loss = validate_step(model, criterion, test_loader)
            test_losses.append(val_loss)

            # Save model checkpoint every 10 epochs
            save_model_checkpoint(model, epoch, result_dir)

            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}')

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving model...")
            torch.save(model.state_dict(), f"{result_dir}/model_interrupted.pth")
            break

    # After training, calculate and save metrics
    mae, mse, r2 = calculate_metrics(test_loader, model)
    metrics_dir = f"{result_dir}/{test_stock_name}"
    save_metrics(metrics_dir, mae, mse, r2)


def train_step(model, criterion, optimizer, X_batch, y_batch):
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def validate_step(model, criterion, test_loader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            val_outputs = model(X_test_batch)
            val_loss += criterion(val_outputs, y_test_batch).item()
    return val_loss / len(test_loader)


def get_stock_names(num_csvs):
    names_50 = ['aal.csv', 'AAPL.csv', 'ABBV.csv', 'AMD.csv', 'amgn.csv', 'AMZN.csv', 'BABA.csv', 'bhp.csv', 'bidu.csv',
                'biib.csv', 'BRK-B.csv', 'C.csv', 'cat.csv', 'cmcsa.csv', 'cmg.csv', 'cop.csv', 'COST.csv', 'crm.csv',
                'CVX.csv', 'dal.csv', 'DIS.csv', 'ebay.csv', 'GE.csv', 'gild.csv', 'gld.csv', 'GOOG.csv', 'gsk.csv',
                'INTC.csv', 'KO.csv', 'mrk.csv', 'MSFT.csv', 'mu.csv', 'nke.csv', 'nvda.csv', 'orcl.csv', 'pep.csv',
                'pypl.csv', 'qcom.csv', 'QQQ.csv', 'SBUX.csv', 'T.csv', 'tgt.csv', 'tm.csv', 'TSLA.csv', 'TSM.csv',
                'uso.csv', 'v.csv', 'WFC.csv', 'WMT.csv', 'xlf.csv']
    names_25 = ['AAPL.csv', 'ABBV.csv', 'AMZN.csv', 'BABA.csv', 'BRK-B.csv', 'C.csv', 'COST.csv', 'CVX.csv', 'DIS.csv',
                'GE.csv', 'INTC.csv', 'MSFT.csv', 'nvda.csv', 'pypl.csv', 'QQQ.csv', 'SBUX.csv', 'T.csv', 'TSLA.csv',
                'WFC.csv', 'KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']
    names_5 = ['KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']

    if num_csvs == 5:
        return names_5
    elif num_csvs == 25:
        return names_25
    elif num_csvs == 50:
        return names_50
    else:
        raise ValueError("[Warning] The number of CSVs is invalid, please choose from 5, 25, or 50.")


def main(data_path, num_csvs, cols_to_scale, cols_to_normalize, test_stock_name, num_epochs, seq_len,
         batch_size, dropout, feature_columns, label_columns, test_size, hidden_size, num_layers, learning_rate,
         use_gpu=True):
    # Directories for saving results
    results_path = f"../src/lstm/test_result_{num_csvs}"
    os.makedirs(results_path, exist_ok=True)

    # Data Processor
    dp = DataProcessor(
        dir_path=data_path,
        seq_len=seq_len,
        test_size=test_size,
        cols_to_scale=cols_to_scale,
        cols_to_normalize=cols_to_normalize,
        batch_size=batch_size,
        feature_columns=feature_columns,
        label_columns=label_columns
    )

    # Define test stock names based on num_csvs
    stock_names = get_stock_names(num_csvs)

    # Check CUDA availability
    Environment.check_cuda(self=Environment(), use_gpu=use_gpu)

    # Prepare data
    dp.csv_to_df(stocks_list=stock_names)
    dp.split_df()
    dp.feature_transformation(dp.train_data)
    dp.feature_transformation(dp.test_data)

    # Get data loaders
    train_loader = dp.get_data_loader(True)
    test_loader = dp.get_data_loader(False)

    # Model parameters
    input_size = next(iter(train_loader))[0].shape[2]
    output_size = next(iter(train_loader))[1].shape[1]

    # Model, Loss, Optimizer
    model = LSTM(input_size, hidden_size, num_layers, output_size, dropout)
    criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Training and validation
    train_and_validate(model, criterion, optimizer, train_loader, test_loader, num_epochs, results_path,
                       test_stock_name=test_stock_name)





if __name__ == "__main__":
    main("../dataset/processed/data_for_lstm",
         5,
         [1, 2, 3, 4, 5, 6],
         [1, 2, 3, 4, 5, 6],
         test_stock_name='GOOG',
         dropout=0.3,
         num_epochs=2000,
         seq_len=60,
         batch_size=32,
         feature_columns=['Open', 'High', 'Low', 'Volume', 'Sentiment_gpt', 'News_flag', 'Scaled_sentiment',
                          'month_sin', 'month_cos', 'day_sin', 'day_cos', 'weekday'],
         label_columns=['Close', 'Adj close'],
         test_size=0.2,
         learning_rate=0.001,
         hidden_size=64,
         num_layers=2,
         use_gpu=True)

