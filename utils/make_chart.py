import pandas as pd
import chart_studio.plotly
import chart_studio.plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.dates as mdate
from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def write_img(profit_list, date_list, symbol_list, title, tick_spacing):
    color_list = ['b', 'c', 'g', 'r', 'm', 'y','k',]
    rcParams['font.sans-serif'] = 'kaiti'
    rcParams['axes.unicode_minus'] = False
    # 创建一个画布
    fig = plt.figure(figsize=(12, 9))
    # 在画布上添加一个子视图
    ax = plt.subplot(111)
    # 这里很重要  需要 将 x轴的刻度 进行格式化
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    # 生成一个时间序列
    time = pd.to_datetime(date_list[0], format='%Y-%m-%d%H:%M')
    # 为X轴添加刻度
    plt.xticks(pd.date_range(time[0], time[-1], freq='D'), rotation=45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    # 设置标题
    ax.set_title(f'资金曲线 Equity Curve {title}')
    # 设置 x y 轴名称
    ax.set_xlabel(f'日期 Date', fontsize=20)
    ax.set_ylabel('净利润 Profit', fontsize=20)
    for index, profit_data in enumerate(profit_list):
        # 生成数据
        data = np.array(profit_data)
        # 画折线
        ax.plot(time, data, color=color_list[index], label=symbol_list[index])
    plt.grid()
    plt.legend()
    plt.show()

def draw_polyline(df, title, y_title):
    import plotly.express as px
    # df = px.data.stocks()
    fig = px.line(df, x="date", y=df.columns,
                  hover_data={"date": "|%B %d, %Y"})
    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y")
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=y_title)
    fig.show()