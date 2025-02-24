import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_processor import DataProcessor
from src.lstm.lstm import LSTM
from src.utils import Environment
from datetime import datetime


current_time = datetime.now().strftime("%Y%m%d%H")


def save_model_checkpoint(model, epoch, result_dir):
    if (epoch + 1) % 50 == 0:
        checkpoint_path = f"{result_dir}/model_checkpoint_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)


def calculate_metrics(test_loader, model, device):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            y_true.append(y_batch)
            y_pred.append(predictions)
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2


def save_metrics(metrics_dir, mae, mse, r2):
    os.makedirs(metrics_dir, exist_ok=True)
    with open(f"{metrics_dir}/metrics.txt", "w") as f:
        f.write(f"MAE: {mae}\nMSE: {mse}\nR2 Score: {r2}\n")


def train_step(model, criterion, optimizer, X_batch, y_batch, device):
    model.train()
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def validate_step(model, criterion, test_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            total_loss += criterion(outputs, y_batch).item()
    return total_loss / len(test_loader)


def train_and_validate(model, criterion, optimizer, train_loader, test_loader, num_epochs, result_dir, test_stock_name,
                       device):
    for epoch in range(num_epochs):
        try:
            epoch_train_losses = []
            for X_batch, y_batch in train_loader:
                loss = train_step(model, criterion, optimizer, X_batch, y_batch, device)
                epoch_train_losses.append(loss)
            train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            val_loss = validate_step(model, criterion, test_loader, device)
            save_model_checkpoint(model, epoch, result_dir)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving model...")
            torch.save(model.state_dict(), f"{result_dir}/model_interrupted.pth")
            break

    mae, mse, r2 = calculate_metrics(test_loader, model, device)
    metrics_dir = f"{result_dir}/{test_stock_name}"
    save_metrics(metrics_dir, mae, mse, r2)


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
    names_1 = ['AMD.csv']

    if num_csvs == 5:
        return names_5
    elif num_csvs == 25:
        return names_25
    elif num_csvs == 50:
        return names_50
    elif num_csvs == 1:
        return names_1
    else:
        raise ValueError("Invalid number of CSVs; choose from 1, 5, 25, or 50.")


def main(data_path, num_csvs, cols_to_normalize, test_stock_name, epochs, seq_len, batch_size, dropout, feature_columns,
         label_columns, test_size, smoothing_window_size, hidden_size, num_layers, learning_rate, use_gpu):

    results_path = f"../src/lstm/test_result_{num_csvs}"
    os.makedirs(results_path, exist_ok=True)

    dp = DataProcessor(
        dir_path=data_path,
        seq_len=seq_len,
        test_size=test_size,
        cols_to_normalize=cols_to_normalize,
        batch_size=batch_size,
        smoothing_window_size=smoothing_window_size,
        feature_columns=feature_columns,
        label_columns=label_columns
    )

    stock_names = get_stock_names(num_csvs)

    env = Environment(use_gpu=use_gpu)
    env.check_cuda()

    # Use the correct method names: load_csv and normalize_features.
    dp.load_csv(stocks_list=stock_names)
    dp.split_df()
    dp.normalize_features()

    train_loader = dp.get_data_loader(is_train=True)
    test_loader = dp.get_data_loader(is_train=False)

    input_size = next(iter(train_loader))[0].shape[2]
    output_size = next(iter(train_loader))[1].shape[1]

    model = LSTM(input_size, hidden_size, num_layers, output_size, dropout)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_and_validate(model, criterion, optimizer, train_loader, test_loader, epochs, results_path, test_stock_name,
                       device)


if __name__ == "__main__":
    main("../dataset/processed/data_for_lstm",
         5,
         ['Open', 'High', 'Low', 'Volume', 'Sentiment_gpt', 'News_flag', 'Scaled_sentiment'],
         test_stock_name='GOOG',
         dropout=0.3,
         epochs=200,
         seq_len=60,
         batch_size=64,
         feature_columns=['Open', 'High', 'Low', 'Volume', 'Sentiment_gpt', 'News_flag', 'Scaled_sentiment'],
         label_columns=['Close'],
         test_size=0.2,
         smoothing_window_size=2500,
         learning_rate=0.001,
         hidden_size=64,
         num_layers=2,
         use_gpu=True)
