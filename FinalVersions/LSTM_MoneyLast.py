import pandas as pd

import datetime

import matplotlib.pyplot as plt  # Отрисовка графиков
import numpy as np  # Numpy

from keras.models import Sequential, Input  # Два варианты моделей
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed,Dropout  # Стандартные слои
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.regularizers import l2
# Выводим параметры файла
data = pd.read_csv(r'D:\!Ilya\MacineLearning\OxaProject\cleaned_costs.csv', index_col=0)
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date')
data = data.fillna(method='ffill')
print(data)
data = data.drop(['Steuer(USD)', 'Date'], axis=1)

scaler = StandardScaler()
data = data.values
print(data.shape)

cutoff_point = int(len(data[:, 0]) * 0.8)
train_data = data[:cutoff_point, ]
test_data = data[cutoff_point + 1:, ]
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


lookback = 10  # если =1, то предсказываем по предыдущему значению
count_of_predict = 2  # количество предсказываемых значений
xTrain, yTrain = split_sequence(train_data, lookback, count_of_predict)
xTest, yTest = split_sequence(test_data, lookback, count_of_predict)

features = 5  # Количество выходных слоёв

### Encoder-Decoder LSTM ###

#todo поиграть с активациями
model = Sequential()
model.add(LSTM(500, input_shape=(lookback, features), activation='sigmoid'))
model.add(RepeatVector(count_of_predict))
model.add(LSTM(500, return_sequences=True, activation='sigmoid'))
model.add(TimeDistributed(Dense(features)))
model.compile(loss='mse', optimizer='adam')
history = model.fit(xTrain, yTrain, epochs=100, validation_data=(xTest, yTest))



plt.plot(history.history['loss'], label='Ошибка на обучающем наборе')
plt.plot(history.history['val_loss'], label='Ошибка на проверочном наборе')
plt.ylabel('Средняя ошибка')
plt.legend()
plt.show()

yhat = model.predict(xTest)


yTest = scaler.inverse_transform(yTest)
yhat = scaler.inverse_transform(yhat)

for i in range(features):
    rmse = sqrt(mean_squared_error(yTest[:, :, i], yhat[:, :, i]))
    plt.title(str(i+1) + ' столбец ' + 'Test RMSE: %.3f' % rmse)
    for j in range(count_of_predict):
        plt.plot(yhat[:, j, i], label='predict' + str(j+1))
    plt.plot(yTest[:, 0, i], label='true')
    plt.legend()
    plt.show()
    for j in range(count_of_predict):
        plt.title(str(i+1) + ' столбец')
        plt.plot(yhat[:, j, i], label='predict'+str(j+1))
        plt.plot(yTest[:, 0, i], label='true'+str(j+1))
        plt.legend()
        plt.show()


    print('Test' + str(i+1) + ' RMSE: %.3f' % rmse)
