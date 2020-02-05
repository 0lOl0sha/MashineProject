import pandas as pd

import datetime

import matplotlib.pyplot as plt  # Отрисовка графиков
import numpy as np  # Numpy

from keras.models import Sequential, Model  # Два варианты моделей
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed  # Стандартные слои
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.regularizers import l2


# Загружаем выборку
def getData(df):
    values = df.values  # Вытаскиваем значение из data frame
    data = []  # Создаём пустую базу

    # Проходим по всем строкам данных и преобразовываем время в удобный формат
    for v in values:
        # Разбиваем на значения, раделитель - ; # Отбрасываем два первых значения - в них даты
        # v[0] = datetime.datetime.strptime(v[0], '%Y-%m-%d %H:%M:%S').timestamp()
        v[1] = datetime.datetime.strptime(v[1], '%Y-%m-%d').timestamp()
        data.append(v)  # Добавляем элемент в базу

    return data


# Получаем данные из файла
def getDataFromFile(fileName):
    df = pd.read_csv(fileName)  # Считываем файл с помощью pandas
    df = df.sort_values('Date')
    # df = df.sort_values('Datetime')
    return getData(df)  # Возвращаем считанные данные из файла


# Выводим параметры файла
data = getDataFromFile(r'D:\!Ilya\MacineLearning\OxaProject\cleaned_costs.csv')
# data = getDataFromFile(r'D:\!Ilya\MacineLearning\OxaProject\COMED_hourly.csv')

d = data
print(len(d))  # Сколько есть записей
data = np.array(data)  # Превращаем в numpy массив


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)


# Препроцессинг данных
for lel in range (5):
    flatX = data[:, 1]
    flatY = data[:, lel+2].reshape(-1, 1)

    cutoff_point = int(len(flatX) * 0.6)

    trainFlatX = flatX[:cutoff_point]
    testFlatX = flatX[cutoff_point + 1:]

    trainFlatY = flatY[:cutoff_point, ]
    testFlatY = flatY[cutoff_point + 1:, ]
    scaler = StandardScaler()
    trainFlatY = scaler.fit_transform(trainFlatY)
    testFlatY = scaler.transform(testFlatY)
    print(len(trainFlatY))
    print(len(testFlatY))


    # Преобразование пайплайна из 2D в 3D для подачи LSTM

    look_back = 10 # количество признаков
    trainX, trainY = create_dataset(trainFlatY, look_back)
    testX, testY = create_dataset(testFlatY, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # trainX = np.reshape(trainX, (trainX.shape[0], 3))
    # testX = np.reshape(testX, (testX.shape[0], 3))

    # Создание модели
    model = Sequential()
    model.add(LSTM(250, input_shape=(trainX.shape[1], trainX.shape[2]), activation='linear', return_sequences=True))
    model.add(LSTM(250, activation='linear', return_sequences=True))
    model.add(LSTM(250, activation='linear', return_sequences=True))
    model.add(LSTM(250, activation='linear'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=20, batch_size=15, validation_data=(testX, testY))
    # history = model.fit(trainX, trainY, epochs=10, batch_size=100, validation_data=(testX, testY))

    # График тренировочной и валидационной выборки
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # Предсказываем значение и сравниваем его, построив график
    yhat = model.predict(testX)

    # Преобразовываем данные в исходный формат для проверки RMSE
    yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
    testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))

    rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))

    plt.title('Test RMSE: %.3f' % rmse)
    plt.plot(yhat_inverse, label='predict')
    plt.plot(testY_inverse, label='true')
    plt.legend()
    plt.show()

# RMSE

    print('Test RMSE: %.3f' % rmse)

# Предсказываем значение для даты, которую модель не видела