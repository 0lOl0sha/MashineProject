import pandas as pd

import datetime

import matplotlib.pyplot as plt  # Отрисовка графиков
import numpy as np  # Numpy
from keras.models import Sequential  # Два варианты моделей
from keras.layers import Dense, LSTM, Dropout  # Стандартные слои
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.regularizers import l2

# Предсказывание нескольких значений
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
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Data preprocessing для 2 модели
d = pd.read_csv(r'D:\!Ilya\MacineLearning\OxaProject\DUQ_hourly.csv')
d['Datetime'] = pd.to_datetime(d['Datetime'])
d.sort_values('Datetime')
d = d.drop(['Datetime'], axis=1)
d = np.array(d)
cutoff_point = int(len(d) * 0.8)
train_data = d[:cutoff_point]
test_data = d[cutoff_point + 1:]

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data).ravel()
test_data = scaler.transform(test_data).ravel()
train_data = list(train_data.ravel())[:]
test_data = list(test_data.ravel())[:]

lookback = 5  # на сколько шагов назад смотрим при предсказании
feature = 1  # входной вектор
count_of_predict = 3  # сколько предсказываем
xTrain, yTrain = split_sequence(train_data, lookback, count_of_predict)
xTest, yTest = split_sequence(test_data, lookback, count_of_predict)
xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], feature))
xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], feature))
# Модель 2
model = Sequential()
model.add(LSTM(25, activation='linear', input_shape=(lookback, feature))) #return_sequences=True))
model.add(Dense(count_of_predict, activation='linear'))
model.compile(optimizer='adam', loss='mse')
history = model.fit(xTrain, yTrain, batch_size=20, epochs=30, validation_data=(xTest, yTest))
model.summary()

yhat = model.predict(xTest, verbose=0)
yTest = scaler.inverse_transform(yTest)
yhat = scaler.inverse_transform(yhat)


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

for i in range(count_of_predict):
    plt.plot(yhat[0:300,i], label='predict'+ str(i+1))
plt.plot(yTest[0:300,0], label='true')
plt.legend()
plt.show()
# Преобразовываем данные в исходный формат для проверки RMSE

for i in range(count_of_predict):
    plt.plot(yhat[0:100,i], label='predict'+ str(i+1))
    plt.plot(yTest[0:100,0], label='true')
    plt.legend()
    plt.show()

rmse = sqrt(mean_squared_error(yTest, yhat))
print('Test RMSE: %.3f' % rmse)