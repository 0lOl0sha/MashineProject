import pandas as pd

import datetime

import matplotlib.pyplot as plt  # Отрисовка графиков
import numpy as np  # Numpy
from keras.models import Sequential  # Два варианты моделей
from keras.layers import Dense, LSTM, Dropout, Conv1D  # Стандартные слои
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.regularizers import l2


#Функция "раскусывания" данных для временных рядов
#data - данные
#xLen - размер фрема, по которому предсказываем
#xChannels - лист, номера каналов, по которым делаем анализ
#yChannels - лист, номера каналов, которые предсказываем
#stepsForward - на сколько шагов предсказываем в будутее
#если 0 - то на 1 шаг, можно использовать только при одном канале, указанном в yChannels
#xNormalization - нормализация входных каналов, 0 - нормальное распределение, 1 - к отрезку [0;1]
#yNormalization - нормализация прогнозируемых каналов, 0 - нормальное распределение, 1 - к отрезку [0;1]
#returnFlatten - делать ли одномерный вектор на выходе для Dense сетей
#valLen - сколько примеров брать для проверочной выборки (количество для обучающей посчитается автоматиески)
#convertToDerivative - bool, преобразовывали ли входные сигналы в производную
def getXTrainFromTimeSeries(data, xLen, xChannels, yChannels, stepsForward, xNormalization, yNormalization, returnFlatten, valLen, convertToDerivative):

  #Если указано превращение данных в производную
  #То вычитаем поточечно из текущей точки предыдущую
  if (convertToDerivative):
    data = np.array([(d[1:]-d[:-1]) for d in data.T]).copy().T

  #Выбираем тип нормализации x
  #0 - нормальное распределение
  #1 - нормирование до отрезка 0-1
  if (xNormalization == 0):
    xScaler = StandardScaler()
  else:
    xScaler = MinMaxScaler()

  #Берём только те каналы, которые указаны в аргументе функции
  xData = data[:,xChannels]
  #Обучаем нормировщик
  xScaler.fit(xData)
  #Нормируем данные
  xData = xScaler.transform(xData)

  #Выбираем тип нормализации y
  #0 - нормальное распределение
  #1 - нормирование до отрезка 0-1
  if (yNormalization == 0):
    yScaler = StandardScaler()
  else:
    yScaler = MinMaxScaler()

  #Берём только те каналы, которые указаны в аргументе функции
  yData = data[:,yChannels]
  #Обучаем нормировщик
  yScaler.fit(yData)
  #Нормируем данные
  yData = yScaler.transform(yData)

  #Формируем xTrain
  #Раскусываем исходный ряд на куски xLen с шагом в 1
  xTrain = np.array([xData[i:i+xLen] for i in range(xData.shape[0]-xLen-1-stepsForward)])

  #Формируем yTrain
  #Берём stepsForward шагов после завершения текущего x
  if (stepsForward > 0):
    yTrain = np.array([yData[i+xLen:i+xLen+stepsForward] for i in range(yData.shape[0]-xLen-1-stepsForward)])
  else:
    yTrain = np.array([yData[i+xLen+stepsForward] for i in range(yData.shape[0]-xLen-1-stepsForward)])

  #Делаем reshape y в зависимости от того
  #Прогнозируем на 1 шаг вперёдили на несколько
  if (stepsForward == 0):
    if ((len(yChannels) == 1)):
      yTrain = yTrain.reshape(yTrain.shape[0], 1)
  else:
      yTrain = yTrain.reshape(yTrain.shape[0], stepsForward)

  #Расчитыываем отступ между обучающими о проверочными данными
  #Чтобы они не смешивались
  xTrainLen = xTrain.shape[0]
  bias = xLen + stepsForward + 2

  #Берём из конечной части xTrain проверочную выборку
  xVal = xTrain[xTrainLen-valLen:]
  yVal = yTrain[xTrainLen-valLen:]
  #Оставшуюся часть используем под обучающую выборку
  xTrain = xTrain[:xTrainLen-valLen-bias]
  yTrain = yTrain[:xTrainLen-valLen-bias]

  #Если в функцию передали вернуть flatten сигнал (для Dense сети)
  #xTrain и xVal превращаем в flatten
  if (returnFlatten > 0):
    xTrain = np.array([x.flatten() for x in xTrain])
    xVal = np.array([x.flatten() for x in xVal])

  return (xTrain, yTrain), (xVal, yVal), (xScaler, yScaler)



def correlate(a, b):
    # РАссчитываем основные показатели
    ma = a.mean()  # Среднее значение первого вектора
    mb = b.mean()  # Среднее значение второго вектора
    mab = (a * b).mean()  # Среднее значение произведения векторов
    sa = a.std()  # Среднеквадратичное отклонение первого вектора
    sb = b.std()  # Среднеквадратичное отклонение второго вектора

    # Рассчитываем корреляцию
    val = 0
    if ((sa > 0) & (sb > 0)):
        val = (mab - ma * mb) / (sa * sb)
    return val


# Функция рисуем корреляцию прогнозированного сигнала с правильным
# Смещая на различное количество шагов назад
# Для проверки появления эффекта автокорреляции
# channels - по каким каналам отображать корреляцию
# corrSteps - на какое количество шагов смещать сигнал назад для рассчёта корреляции
def showCorr(channels, corrSteps, predVal, yValUnscaled):
    # Проходим по всем каналам
    for ch in channels:
        corr = []  # Создаём пустой лист, в нём будут корреляции при смезении на i рагов обратно
        yLen = yValUnscaled.shape[0]  # Запоминаем размер проверочной выборки

        # Постепенно увеличикаем шаг, насколько смещаем сигнал для проверки автокорреляции
        for i in range(corrSteps):
            # Получаем сигнал, смещённый на i шагов назад
            # predVal[i:, ch]
            # Сравниваем его с верными ответами, без смещения назад
            # yValUnscaled[:yLen-i,ch]
            # Рассчитываем их корреляцию и добавляем в лист
            corr.append(correlate(yValUnscaled[:yLen - i, ch], predVal[i:, ch]))

        # Отображаем график коррелций для данного шага
        plt.plot(corr, label='предсказание на ' + str(ch + 1) + ' шаг')

    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.show()


# Загружаем выборку
def getData(df):
    df.index = pd.to_datetime(df.index)
    values = df.values  # Вытаскиваем значение из data frame
    data = []  # Создаём пустую базу

    # Проходим по всем строкам данных и преобразовываем время в удобный формат
    for v in values:
        # Разбиваем на значения, раделитель - ; # Отбрасываем два первых значения - в них даты
        # v[0] = datetime.datetime.strptime(v[0], '%Y-%m-%d %H:%M:%S').timestamp()
        data.append(v)  # Добавляем элемент в базу

    return data


# Получаем данные из файла
def getDataFromFile(fileName):
    df = pd.read_csv(fileName)  # Считываем файл с помощью pandas
    df = df.sort_values('Datetime')
    return getData(df)  # Возвращаем считанные данные из файла





# Метод окна
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

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

d = pd.read_csv(r'D:\!Ilya\MacineLearning\OxaProject\PJME_hourly.csv')
d['Datetime'] = pd.to_datetime(d['Datetime'])
d.sort_values('Datetime')
d = np.array(d)
#d = d.drop(['Datetime'], axis=1)

lookback = 30 # на сколько шагов назад смотрим при предсказании
feature = 1  # входной вектор
count_of_predict = 3  # сколько предсказываем

(xTrain, yTrain), (xTest, yTest), (xScaler, yScaler) = getXTrainFromTimeSeries(d, lookback, [1], [1],
                                                                             count_of_predict, 0,
                                                                             0, 0, int(len(d)*0.2),
                                                                             0)

# Модель 2
model = Sequential()
model.add(LSTM(30, activation='linear', input_shape=(lookback, feature)))
# model.add(LSTM(8))
model.add(Dense(count_of_predict, activation='linear'))
model.compile(optimizer='adam', loss='mse')
history = model.fit(xTrain, yTrain, batch_size=100, epochs=10, validation_data=(xTest, yTest))
# денежный решейп,
# xTest = xTest.reshape(1, lookback, feature)
yhat = model.predict(xTest, verbose=0)
yTest = yScaler.inverse_transform(yTest)
yhat = yScaler.inverse_transform(yhat)
# print(yTest)
# print(yhat)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
"""
for i in range(count_of_predict):
    plt.plot(yhat[0:300, i], label='predict' + str(i + 1))
plt.plot(yTest[0:300, 0], label='true')
plt.legend()
plt.show()
# Преобразовываем данные в исходный формат для проверки RMSE


plt.plot(yhat[0:20, 0], label='predict' + str(1))
plt.plot(yhat[0:20, 1], label='predict' + str(2))
plt.legend()
plt.show()

for i in range(count_of_predict):
    plt.plot(yhat[0:300, i], label='predict' + str(i + 1))
    plt.plot(yTest[0:300, i], label='true')
    plt.legend()
    plt.show()
"""
rmse = sqrt(mean_squared_error(yTest, yhat))
print('Test RMSE: %.3f' % rmse)

for i in range(count_of_predict):
    showCorr([i], lookback, yhat, yTest)


