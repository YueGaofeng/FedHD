import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import random


def plot_figure(x, kde_values, parameters, round, id):
    double_column = 2.2
    font28 = {
    'family': 'Arial',
    'weight': 'normal',
    'size': 28 * double_column,
    }
    font24 = {
    'family': 'Arial',
    'weight': 'normal',
    'size': 24 * double_column,
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.plot(x, kde_values, '-', linewidth=2.0*double_column, color='blue', label='LOD')
    plt.hist(parameters, color='white', edgecolor='black', linewidth=1.5*double_column, \
             bins=20, density=True, alpha=0.5, label='Histogram')  # 绘制直方图作为对比

    plt.xlabel('Parameter value', font24)
    plt.ylabel('Density', font24)
    plt.title(id + ' in Round '+str(round), font24)

    # 图例设置
    # 图例设置
    ax.legend(prop = font24, loc='best', labelspacing=0.3, handletextpad=0.3)

    # 增加坐标轴宽度
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.xticks(np.arange(-0.2, max(x), 0.1), size=24*double_column)
    plt.yticks(np.arange(0, max(kde_values)+1, 2), size=24*double_column)
 
    # 保存和展示图形
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.margins(0.0)
    plt.tight_layout(pad=0.1)
    rootpath = './Results'
    plt.savefig(rootpath + '/fig9_{}_{}.png'.format(round, id), format="png", bbox_inches='tight', pad_inches=0)



def plot_figure_global(parameters, round, id):
    kde = gaussian_kde(parameters)
    x = np.linspace(min(parameters), max(parameters), num=50)
    kde_values = kde(x)
    
    double_column = 2.2
    font28 = {
    'family': 'Arial',
    'weight': 'normal',
    'size': 28 * double_column,
    }
    font24 = {
    'family': 'Arial',
    'weight': 'normal',
    'size': 24 * double_column,
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.plot(x, kde_values, '-', linewidth=2.0*double_column, color='blue', label='LOD')
    plt.hist(parameters, color='white', edgecolor='black', linewidth=1.5*double_column, \
             bins=20, density=True, alpha=0.5, label='Histogram')  # 绘制直方图作为对比

    plt.xlabel('Parameter value', size=24*double_column)
    plt.ylabel('Density', size=24*double_column)
    plt.title(id + ' in Round '+str(round), size=24*double_column)

    # 图例设置
    # 图例设置
    ax.legend(prop = font24, loc='best', labelspacing=0.3, handletextpad=0.3)

    # 增加坐标轴宽度
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.xticks(np.arange(-0.2, max(x), 0.1), size=24*double_column)
    plt.yticks(np.arange(0, max(kde_values)+1, 2), size=24*double_column)
 
    # 保存和展示图形
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.margins(0.0)
    plt.tight_layout(pad=0.1)
    rootpath = './Results'
    plt.savefig(rootpath + '/fig9_{}_{}.png'.format(round, id), format="png", bbox_inches='tight', pad_inches=0)

