import os
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from data_processor import DataLoader
from lstm_modified import Model

current_time = datetime.now().strftime('%d%m%Y-%H%M%S')


def output_results_and_errors_multiple(predicted_data, true_data, true_data_base, prediction_len, file_name,
                                       sentiment_type, num_csvs):
    """
    Output the predicted and true values
    :param predicted_data: the predicted data
    :param true_data: the true data
    :param true_data_base: the true data base
    :param file_name: the file name
    :param sentiment_type: the sentiment type
    :param num_csvs: the number of csvs
    """

    # Create an empty DataFrame
    save_df = pd.DataFrame()

    # Add the true data to the DataFrame
    save_df['True_Data'] = true_data.reshape(-1)
    save_df['Base'] = true_data_base.reshape(-1)

    # Convert back to the original scale
    save_df['True_Data_origin'] = (save_df['True_Data'] + 1) * save_df['Base']

    # Concatenate all predicted data together
    if predicted_data:
        all_predicted_data = np.concatenate([p for p in predicted_data])
    else:
        all_predicted_data = predicted_data

    file_name = file_name.split('.')[0]
    sentiment_type = str(sentiment_type)

    # Add the concatenated predicted data to the DataFrame
    save_df['Predicted_Data'] = pd.Series(all_predicted_data)

    # Convert back to the original scale
    save_df['Predicted_Data_origin'] = (save_df['Predicted_Data'] + 1) * save_df['Base']

    # If the length of the predicted values is different, fill with NaN
    save_df = save_df.fillna(np.nan)
    result_folder = f'test_result_{num_csvs}'
    save_file_path = os.path.join(result_folder, f'{file_name}_{sentiment_type}_{current_time}',
                                  f'{file_name}_{sentiment_type}_{current_time}_predicted_data.csv')

    # Save the DataFrame to a CSV file
    os.makedirs(os.path.join(result_folder, f'{file_name}_{sentiment_type}_{current_time}'), exist_ok=True)

    save_df.to_csv(save_file_path, index=False)
    print(f'Data saved to {save_file_path}')
    # Output eval
    # Truncate data to ensure lengths are consistent
    min_length = min(len(save_df['Predicted_Data']), len(save_df['True_Data']))
    predicted_data = save_df['Predicted_Data'][:min_length]
    predicted_data.fillna(method='ffill', inplace=True)
    true_data = save_df['True_Data'][:min_length]

    # Calculate MAE, MSE, R²
    mae = mean_absolute_error(true_data, predicted_data)
    mse = mean_squared_error(true_data, predicted_data)
    r2 = r2_score(true_data, predicted_data)

    results_df = pd.DataFrame({
        'MAE': [mae],
        'MSE': [mse],
        'R2': [r2]
    })

    eval_file_path = os.path.join(result_folder, f'{file_name}_{sentiment_type}_{current_time}',
                                  f'{file_name}_{sentiment_type}_{current_time}_eval.csv')

    # Save the eval DataFrame to a CSV file
    results_df.to_csv(eval_file_path, index=False)
    print(f'Eval saved to {eval_file_path}')


def main(configurations, data_filename, sentiment, flag_pred, model, num_csvs):
    """
    Main function
    :param configurations: the configurations
    :param data_filename: the data filename
    :param sentiment: the sentiment type
    :param flag_pred: the prediction flag
    :param model: the model name
    :param num_csvs: the number of csvs
    """

    symbol_name = data_filename.split('.')[0]
    if not os.path.exists(configurations['model']['save_dir']):
        os.makedirs(configurations['model']['save_dir'])

    data = DataLoader(
        os.path.join('../data', data_filename),
        configurations['data']['train_test_split'],
        configurations['data']['columns'],
        configurations['data']['columns_to_normalise'],
        configurations['data']['sequence_length']
    )

    model = Model()
    model_path = f'saved_models/{model}_{sentiment}_{num_csvs}.h5'
    if os.path.exists(model_path):
        model.load_model(model_path)
    else:
        model.build_model(configurations)

    x, y = data.get_train_data(
        seq_len=configurations['data']['sequence_length'],
        normalise=configurations['data']['normalise']
    )
    print("X:", x.shape)
    print("Y:", y.shape)
    # out-of memory generative training
    steps_per_epoch = math.ceil(
        (data.len_train - configurations['data']['sequence_length']) / configurations['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configurations['data']['sequence_length'],
            batch_size=configurations['training']['batch_size'],
            normalise=configurations['data']['normalise']
        ),
        epochs=configurations['training']['epochs'],
        batch_size=configurations['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configurations['model']['save_dir'],
        sentiment_type=sentiment,
        model_name=model_name,
        num_csvs=num_csvs
    )
    if flag_pred:
        if symbol_name in pred_names:
            print(f'------Predicting {symbol_name}------')
            x_test, y_test, y_base = data.get_test_data(
                seq_len=configurations['data']['sequence_length'],
                normalise=configurations['data']['normalise'],
                cols_to_norm=configurations['data']['columns_to_normalise']
            )
            print("test data:\nX: ", x_test.shape, "\nY: ", y_test.shape)
            predictions = model.predict_sequences_multiple(x_test, configurations['data']['sequence_length'],
                                                           configurations['data']['sequence_length'])

            output_results_and_errors_multiple(predictions, y_test, y_base, configs['data']['prediction_length'],
                                               symbol_name, sentiment_type, num_csvs)


if __name__ == '__main__':
    model_name = 'LSTM'
    sentiment_types = ["sentiment"]  # or "nonsentiment"

    # Test csvs = 50
    names_50 = ['aal.csv', 'AAPL.csv', 'ABBV.csv', 'AMD.csv', 'amgn.csv', 'AMZN.csv', 'BABA.csv', 'bhp.csv', 'bidu.csv',
                'biib.csv', 'BRK-B.csv', 'C.csv', 'cat.csv', 'cmcsa.csv', 'cmg.csv', 'cop.csv', 'COST.csv', 'crm.csv',
                'CVX.csv', 'dal.csv', 'DIS.csv', 'ebay.csv', 'GE.csv', 'gild.csv', 'gld.csv', 'GOOG.csv', 'gsk.csv',
                'INTC.csv', 'KO.csv', 'mrk.csv', 'MSFT.csv', 'mu.csv', 'nke.csv', 'nvda.csv', 'orcl.csv', 'pep.csv',
                'pypl.csv', 'qcom.csv', 'QQQ.csv', 'SBUX.csv', 'T.csv', 'tgt.csv', 'tm.csv', 'TSLA.csv', 'TSM.csv',
                'uso.csv', 'v.csv', 'WFC.csv', 'WMT.csv', 'xlf.csv']

    # Test csvs = 25
    names_25 = ['AAPL.csv', 'ABBV.csv', 'AMZN.csv', 'BABA.csv', 'BRK-B.csv', 'ebay.csv', 'C.csv', 'COST.csv', 'CVX.csv',
                'DIS.csv', 'GE.csv', 'INTC.csv', 'MSFT.csv', 'nvda.csv', 'pypl.csv', 'QQQ.csv', 'SBUX.csv', 'T.csv',
                'TSLA.csv', 'WFC.csv', 'KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']

    # Test csvs = 5
    names_5 = ['KO.csv', 'AMD.csv', 'TSM.csv', 'GOOG.csv', 'WMT.csv']

    all_names = [names_5, names_25, names_50]
    pred_names = ['KO', 'AMD', "TSM", "GOOG", 'WMT']

    for names in all_names:
        num_stocks = len(names)
        for i in range(3):
            if_pred = False
            if i == 2:
                if_pred = True
            for sentiment_type in sentiment_types:
                for name in names:
                    print(name)
                    configs = json.load(open(sentiment_type + '_config.json', 'r'))
                    main(configs, name, sentiment_type, if_pred, model_name, num_stocks)
