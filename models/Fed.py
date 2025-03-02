#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# FedAvg

import copy
import torch
from converter import Model_Distribution, avg_polys, Distribution_Model

def FedAvg_cifar(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg(w, tau):
    w_avg = copy.deepcopy(w[0])
    
    for k in w_avg.keys():
        poly_list = []
        para_list = []
        for i in range(0, len(w)):
            if k == 'fc2.weight':
                local_poly, kde_values, parameters = Model_Distribution(copy.deepcopy(w[i][k]), fit_degree=100, \
                                                            round=tau, label = 'Client '+str(i+1))
                poly_list.append(local_poly)
                para_list.append(parameters)
            else:
                w_avg[k] += w[i][k]

        if k == 'fc2.weight':
            avg_poly = avg_polys(poly_list)
            parameters = calculate_average_of_lists(para_list)
            w_avg_ds = Distribution_Model(avg_poly, kde_values, parameters, copy.deepcopy(w_avg[k]), round=tau)
            w_avg[k] = w_avg_ds
        else: w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg


def calculate_average_of_lists(list_of_lists):
    # 初始化一个长度为子列表长度的列表，用于存储每个位置上子列表的平均值
    avg_list = [0] * len(list_of_lists[0])

    # 计算每个位置上子列表的平均值
    for sublist in list_of_lists:
        for i in range(len(sublist)):
            avg_list[i] += sublist[i]

    # 求每个位置上子列表的平均值
    avg_list = [elem / len(list_of_lists) for elem in avg_list]

    return avg_list
