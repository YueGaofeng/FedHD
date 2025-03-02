import matplotlib,torch
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sympy import symbols
from drawing import plot_figure,plot_figure_global
import random


def normalize_values(values):
    min_val = min(values)
    max_val = max(values)
    normalized_values = [(2*(val - min_val)/(max_val - min_val) - 1) for val in values]
    return normalized_values

#  已知y值和表达式系数列表， 求多项式对应的x值
def solve_polynomial_values(y_values, coefficients):
    # 创建多项式函数
    def polynomial_function(x, coefficients):
        y = 0
        for i in range(len(coefficients)):
            y += coefficients[i] * (x ** i)
        return y
    # 找到多项式y值对应的x值
    x_values = []
    for y in y_values:
        x_root = np.roots(np.array(coefficients) - y)
        real_roots = x_root[np.isreal(x_root)].real
        real_roots = sorted(real_roots, key=lambda x: abs(x - 0.5))  # 将根按照距离0.5的远近排序
        x_values.append(real_roots[0])  # 选择0.5作为基准值是一种常见的做法，因为这个值可以表示中间位置，对称性和平衡性。
    return x_values


def Model_Distribution(parameters, fit_degree, round, label):
    Array = parameters.numpy()
    parameters = Array.flatten().tolist()
    kde = gaussian_kde(parameters)
    x = np.linspace(min(parameters), max(parameters), fit_degree)
    kde_values = kde(x)
    local_poly = np.polyfit(x, kde_values, deg=49)  # 这里选择50次多项式进行拟合
    plot_figure(x, kde_values, parameters, round, label)
    return local_poly, kde_values,parameters

# 多项式求和加平均
def avg_polys(locals_poly):
    average_values = [sum(item) / len(item) for item in zip(*locals_poly)]
    return average_values

# 分布求参数(MLE)
def Distribution_Model(avg_poly, y_values, parameters, torch_w, round):
    # increase rubust for FL
    parameters = [num +  random.uniform(0, 0.05) for num in parameters]
    # 计算所有元素的总数
    total_elements = torch_w.numel()
    y_virtual = np.random.normal(min(y_values), max(y_values), total_elements)
    x_values = solve_polynomial_values(y_virtual, avg_poly)
    tensor_y = torch.tensor(parameters).view(torch_w.shape)
    #tensor_y = torch.tensor(y_virtual).view(torch_w.shape)
    kde = gaussian_kde(x_values)
    x = np.linspace(min(x_values), max(x_values), 100)
    kde_values = kde(x)
    local_poly = np.polyfit(x, kde_values, deg=49)  # 这里选择50次多项式进行拟合
    #plot_figure_global(x, kde_values, parameters, round, id='Gobal')
    plot_figure_global(parameters, round, id='Global')

    return tensor_y

