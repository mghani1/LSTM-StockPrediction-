import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import preprocessing

def get_data(prices, fundamentals):

    df = pd.read_csv(prices, index_col = 0)
    df = df[['symbol', 'open', 'low', 'high', 'volume', 'close']]
 
    symbols = list(set(df.symbol))

    train_input = []
    train_labels = []
    test_input = []
    test_labels = []

    for symbol in symbols:

        stock = df[df.symbol == symbol].copy()
        data = stock.values 
        num_days = len(data)
        if num_days != 1762:
            continue 

        stock.drop(['symbol'], 1, inplace=True)

        min_max_scaler = preprocessing.MinMaxScaler()
        stock['open'] = min_max_scaler.fit_transform(stock.open.values.reshape(-1, 1))
        stock['high'] = min_max_scaler.fit_transform(stock.high.values.reshape(-1, 1))
        stock['low'] = min_max_scaler.fit_transform(stock.low.values.reshape(-1, 1))
        stock['volume'] = min_max_scaler.fit_transform(stock.volume.values.reshape(-1, 1))
        stock['close'] = min_max_scaler.fit_transform(stock.close.values.reshape(-1, 1))
        
        num_features = 5
        window = 20 
        new_data = []
        
        for i in range(num_days - window): 
            new_data.append(data[i:i + window]) 

        new_data = np.array(new_data)
        limit = round(num_days * 0.9)

        train_input.append(new_data[:limit, :][:, :-1])
        train_labels.append(new_data[:limit, :][:, -1][:, -1])
        test_input.append(new_data[limit:, :][:, :-1])
        test_labels.append(new_data[limit:, :][:, -1][:, -1])

    train_input = np.array(train_input)
    train_labels = np.array(train_labels)
    test_input = np.array(test_input)
    test_labels = np.array(test_labels)

    return train_input, train_labels, test_input, test_labels
