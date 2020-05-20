from random import gauss
import random
from random import seed
from pandas import Series
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import math
import scipy.signal
from scipy.stats import levy_stable
from scipy.signal import wiener
import wave
from scipy.io import wavfile
import numpy as np
import pyaudio
import numpy as np
from PIL import Image

def apply_wiener(my_input, w, order):
    sig = np.zeros(order+np.size(my_input) - 1)
    sig[np.size(sig)-np.size(my_input):np.size(sig)] = my_input
    my_output = np.zeros(np.size(my_input))
    for i in range(np.size(my_input)):
        my_output[i] = np.dot(sig[i:i+order], w)
    return my_output

def error_power(d, my_output):
    error = d-my_output
    return np.sum(error*error)

def update_w(w, sig, d, miu):
    sig = sig.astype(np.float)
    e = d - np.sum(w * sig)
    w = w + 2 * miu * e * sig
    return w, e

def lms(my_input, d, order, miu):
    length = np.size(d)
    # w = np.random.rand(order)
    w = np.zeros(order)
    my_input = np.hstack((np.zeros(order - 1), my_input))
    e_list = []
    mse = []
    mse1 = []
    w_list = []
    for i in range(length):
        e = d[i] - w @ my_input[i:i + order]
        e_list.append(e ** 2)
        if(i % 1000 == 0 or i == length-1):
            mse1.append(np.mean(e_list))
        mse.append(np.mean(e**2))
        w += 2 * miu * e * my_input[i:i + order]
        w_list.append(w)
    return w_list, mse, mse1

def noise_lms(my_input, d, order, miu):
    length = np.size(d)
    # w = np.random.rand(order)
    w = np.zeros(order)
    my_input = np.hstack((np.zeros(order - 1), my_input))
    e_list = []
    mse = []
    mse1 = []
    w_list = []
    for i in range(length):
        e = d[i] - w @ my_input[i:i + order]
        e_list.append(e ** 2)
        if (i % 1000 == 0):
            mse1.append(np.mean(e_list))
        mse.append(np.mean(e ** 2))
        w += 2 * miu * e * my_input[i:i + order]
        w_list.append(w)
    return w_list, mse, mse1

""" wiener filter """
def correlation_matrix(r):
    m = np.zeros((len(r), len(r)))
    for i in range(0, len(r)):
        m[i,i] = 0
        j = i
        while(j-1 >= 0):
            m[i,j-1] = int(m[i,j] + 1)
            #m[i,j-1] = r[m[i,j-1]]
            j = j - 1
        j = i
        while(j+1 < len(r)):
            m[i,j+1] = int(m[i,j] + 1)
            #m[i, j - 1] = r[m[i, j + 1]]
            j = j + 1
    for i in range(0, len(r)):
        for j in range(0, len(r)):
            m[i,j] = r[int(m[i,j])]
    return m

def wiener_filter(order, signal_1, signal_2):#signal_1 is input
    auto = my_corr(signal_1,signal_1, order)
    auto = auto[0:order]
    cross = my_corr(signal_2, signal_1, order)
    cross = cross[0:order]
    test = correlation_matrix(auto)
    w = np.linalg.inv(test) @ cross.T
    return w

def wiener(x, d, order):
    r = my_corr(x, x, order)
    p = my_corr(x, d, order)
    m_r = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            m_r[i, j] = r[abs(i - j)]
    return np.linalg.inv(m_r) @ p

def my_corr(my_input1, my_input2, size):
    my_input1 = my_input1.astype(np.float)
    my_input2 = my_input2.astype(np.float)
    # size = np.size(my_input1)
    r = np.zeros(size)
    r[0] = np.mean(my_input1 * my_input2)
    for i in range(1, size):
        r[i] = np.mean(my_input1[i:] * my_input2[:-i])
    return r

def predict_linear(x, w):
    order = w.size
    n_sample = x.size
    xx = np.zeros((n_sample, order))
    xx[:, 0] = x
    for i in range(1, order):
        xx[i:, i] = x[:-i]
    return xx @ w

def average(x, order):
    length = x.size
    x = np.hstack((x[:order - 1], x))
    r = np.zeros(length)
    for i in range(length):
        r[i] = np.mean(x[i: i + order])
    return r

def gaussian(x, mean, std):
    return 1 / np.sqrt(2 * np.pi) / std * np.exp(-(x - mean) ** 2 / std ** 2 / 2)

def predict_linear_normalized(x, w):
    order = w.size
    n_sample = x.size
    xx = np.zeros((n_sample, order))
    xx[:, 0] = x
    for i in range(1, order):
        xx[i:, i] = x[:-i]
    return xx @ w, np.mean(xx ** 2, 1)