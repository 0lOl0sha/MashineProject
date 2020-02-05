import pandas as pd
from plotly.offline import plot_mpl, iplot
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import cufflinks as cf


#TODO поиграть с параметрами посмотреть что бует

cf.go_offline()
cf.set_config_file(offline=True, world_readable=True)

for i in range (5):
    data = pd.read_csv(r'D:\!Ilya\MacineLearning\FinalVersions\cleaned_costs.csv', index_col=1)
    data = data.drop(['a'], axis=1)
    data = data.drop(['Steuer(USD)'], axis=1)

    if i != 0:
        data = data.drop(['EC2-Instances(USD)'], axis=1)
    if i != 1:
        data = data.drop(['EC2-Andere(USD)'], axis=1)
    if i != 2:
        data = data.drop(['EC2-ELB(USD)'], axis=1)
    if i != 3:
        data = data.drop(['S3(USD)'], axis=1)
    if i != 4:
        data = data.drop(['Gesamtkosten (USD)'], axis=1)


    data.head()
    data.index = pd.to_datetime(data.index)

    data = data.sort_values('Date')

    cutoff_point = int(len(data) * 0.8)

    train = data[:cutoff_point]
    test = data[cutoff_point + 1:]

    from pmdarima.arima import auto_arima

    stepwise_model = auto_arima(data, start_p=1, start_q=1,
                            max_p=3, max_q=3, m=12,
                            start_P=0, seasonal=True,
                            d=1, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
    print(stepwise_model.aic())

    stepwise_model.fit(train)

    future_forecast = stepwise_model.predict(n_periods=test.shape[0])
    print(future_forecast)
    future_forecast = pd.DataFrame(future_forecast, index=test.index)
    if i == 0:
        plt.title('EC2-Instances(USD)')
    if i == 1:
        plt.title('EC2-Andere(USD)')
    if i == 2:
        plt.title('EC2-ELB(USD)')
    if i == 3:
        plt.title('S3(USD)')
    if i == 4:
        plt.title('Gesamtkosten (USD)')

    plt.plot(future_forecast,label='Prediction')
    plt.plot(test,label='True')
    plt.legend()
    plt.show()

    result = seasonal_decompose(data, model='multiplicative',freq=10)
    fig = result.plot()
    plot_mpl(fig)
