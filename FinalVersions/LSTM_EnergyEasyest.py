import pandas as pd

import datetime

import matplotlib.pyplot as plt  # Отрисовка графиков
import numpy as np  # Numpy

from keras.models import Sequential, Model  # Два варианты моделей
from keras.layers import Dense, LSTM,Dropout  # Стандартные слои
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.regularizers import l2


"""
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
  xTrain = np.array([xData[i:i+xLen, xChannels] for i in range(xData.shape[0]-xLen-1-stepsForward)])
  
  #Формируем yTrain
  #Берём stepsForward шагов после завершения текущего x
  if (stepsForward > 0):
    yTrain = np.array([yData[i+xLen:i+xLen+stepsForward, yChannels] for i in range(yData.shape[0]-xLen-1-stepsForward)])
  else:
    yTrain = np.array([yData[i+xLen+stepsForward, yChannels] for i in range(yData.shape[0]-xLen-1-stepsForward)])

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

  return (xTrain, yTrain), (xVal, yVal)

"""

"""
попробовать обыить на одном, тестировать на втором, что будет
поиграть с эпохами
"""
# Загружаем выборку
def getData(df):
    values = df.values  # Вытаскиваем значение из data frame
    data = []  # Создаём пустую базу

    # Проходим по всем строкам данных и преобразовываем время в удобный формат
    for v in values:
        # Разбиваем на значения, раделитель - ; # Отбрасываем два первых значения - в них даты
        v[0] = datetime.datetime.strptime(v[0], '%Y-%m-%d %H:%M:%S').timestamp()

        data.append(v)  # Добавляем элемент в базу

    return data


# Получаем данные из файла
def getDataFromFile(fileName):
    df = pd.read_csv(fileName)  # Считываем файл с помощью pandas

    df = df.sort_values('Datetime')
    return getData(df)  # Возвращаем считанные данные из файла


# Выводим параметры файла

data = getDataFromFile(r'D:\!Ilya\MacineLearning\OxaProject\COMED_hourly.csv')

d = data
print(len(d))  # Сколько есть записей
data = np.array(data)  # Превращаем в numpy массив

# Препроцессинг данных
flatX = data[:, 0]
flatY = data[:, 1].reshape(-1, 1)

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
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)


look_back = 10  # количество признаков
trainX, trainY = create_dataset(trainFlatY, look_back)
testX, testY = create_dataset(testFlatY, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# Создание модели
model = Sequential()
model.add(LSTM(4, input_shape=(trainX.shape[1], trainX.shape[2]), activity_regularizer=l2(0.0)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

history = model.fit(trainX, trainY, epochs=10, batch_size=100, validation_data=(testX, testY))

# График тренировочной и валидационной выборки
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Предсказываем значение и сравниваем его, построив график
yhat = model.predict(testX)

lel = model.predict(testX[0:10])

# Преобразовываем данные в исходный формат для проверки RMSE
yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
plt.plot(yhat_inverse[0:300], label='predict')
plt.plot(testY_inverse[0:300], label='true')
plt.legend()
plt.show()

# RMSE
rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
print('Test RMSE: %.3f' % rmse)

# Предсказываем значение для даты, которую модель не видела
