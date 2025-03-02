#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
from pyparsing import originalTextFor
from sympy import comp
import torch
import random
import math
import numpy as np


def normalize(ls): 
    s = sum(ls) 
    return [i/s for i in ls] 


def compensation(w_start, w_end, Big_gamma):
    ws_diff = copy.deepcopy(w_start)
    we_diff = copy.deepcopy(w_end)
    for k in ws_diff.keys():
        ws_diff[k] = torch.div(ws_diff[k] - we_diff[k], Big_gamma)  # diff-value
    return ws_diff

def add_wgt(wgt_upload, compen):
    wgt_upload = copy.deepcopy(wgt_upload)
    compen = copy.deepcopy(compen)
    for k in wgt_upload.keys():
        wgt_upload[k] = wgt_upload[k] + compen[k]
    return wgt_upload
    
def sub_mean(w1, w2):
    s = 0
    w3 = copy.deepcopy(w2)
    for k in w1.keys():
        w3[k] = abs(w1[k]-w2[k])  # L-1 norm
    s = torch.mean(w3[k])
    return s    

def wgt_time_agg(wgt_metric, rad_index, time_seq, compen, Big_gamma):
    wgt_upload = copy.deepcopy(wgt_metric[rad_index[0]])
    last_wgt = copy.deepcopy(wgt_metric[Big_gamma-1])
    delta = 0.25  # delta is adjustable constant.
    for j in range(len(rad_index)):   # each t time
        wgt_tmp = copy.deepcopy(wgt_metric[rad_index[j]])
        w_key = copy.deepcopy(wgt_tmp[0])
        for C_i in range(0, len(wgt_tmp)):
            for k in w_key.keys():
                wgt_tmp[C_i][k] = wgt_tmp[C_i][k] * time_seq[j]  # resolve time seq
                # Multiple seq-para elements will be add.
                if j == 0: wgt_upload[C_i][k] = wgt_tmp[C_i][k]
                else: wgt_upload[C_i][k] = wgt_upload[C_i][k] + wgt_tmp[C_i][k]      # calculative sum
                if j == (len(rad_index)-1):
                    wgt_upload[C_i][k] = last_wgt[C_i][k]*(1-2*delta) - delta*wgt_upload[C_i][k]\
                        + delta*compen[C_i][k] 
                    #wgt_upload[C_i][k] = wgt_upload[C_i][k] + compen[C_i][k] 
    return wgt_upload


def DRR(w_glob, w_local, reward_client, m_total, m_sub):
    d_subset = 3 # we will randomly pickup d client-subsets
    index = np.zeros((d_subset, m_sub)).astype(np.int)
    con_p_i = []*d_subset  # the contribution of each client across all client
    a_t_d = np.zeros((d_subset, m_sub))
    sita = 0.5
    for d in range(d_subset):
        index[d,:] = random.sample(range(0, m_total), m_sub) # get random subset
        ran_pro =[]           # Contribution value of an arbitrary subset
        for k in index[d]:
            s = sub_mean(copy.deepcopy(w_glob), copy.deepcopy(w_local[k]))
            L1_norm = math.exp(float(s)*100)  # e^x-function to compress the y-value
            ran_pro.append(L1_norm)
        con_p_i = np.array(ran_pro)/sum(ran_pro) # represents the variability of local and global
        for i in range(0, m_sub):
            tmp = 1/(1 + math.exp(0 - 100 * con_p_i[i])) # Sigmoid as in Equation 19
            a_t_d[d, i] = tmp
    sort_id =  np.argsort(list(np.sum(a_t_d, axis=1)))
    se_id = int(sort_id[2])
    Real_client = list(index[se_id])       # choice max information
    wgt_client = reward_client
    #print('selected id:', se_id, list(np.sum(a_t_d, axis=1)))
    for i in range(m_total):
        for j in range(d_subset):      
            if i == Real_client[j]:
                wgt_client[i] = math.pow(sita, a_t_d[se_id, j] - 1) + 1   # add 1 while selected
                reward_client[i] = reward_client[i] + wgt_client[i]  # update reward  
            else:wgt_client[i] = math.pow(sita, reward_client[i] - 1) + 0
    wgt_client = normalize(wgt_client)
    w_avg = copy.deepcopy(w_local[0])
    for k in w_avg.keys():
        for i in range(m_total):
            for j in range(d_subset):      
                if i == Real_client[j]: # for weight
                    w_local[i][k] = torch.mul(w_local[i][k], wgt_client[Real_client[j]])
                else:
                    w_local[i][k] = torch.mul(w_local[i][k], 0)
            w_avg[k] += w_local[i][k]
    return w_avg, normalize(reward_client)