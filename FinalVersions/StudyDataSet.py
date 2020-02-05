import pandas as pd
import matplotlib.pyplot as plt
# Стандартные импорты plotly
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks

cufflinks.go_offline()
# Устанавливаем глобальную тему
cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)

df = pd.read_csv(r'D:\!Ilya\MacineLearning\OxaProject\DUQ_hourly.csv', index_col=0, parse_dates=True)

df = df.reset_index()
fig = go.Figure(go.Scatter(x=df['Datetime'], y=df['DUQ_MW']))
fig.show()
