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
from func import *

"""main"""
[a, orignal_train] = wavfile.read('train_signal.wav')
start_split = [96000, 205158, 314994, 424155, 537225, 647764, 757664, 867693, 984366, 1096043]
end_split = [109158, 218994, 328155, 441225, 551764, 661664, 771693, 888366, 1000043, 1107446]
#starttime: [ 2.002       4.27839913  6.56893738  8.84539906 11.20337969 13.50857842, 15.80045133 18.09501444 20.52813263 22.8570634 ]
#endtime: [ 2.27639913  4.56693738  6.84339906  9.20137969 11.50657842 13.79845133, 16.09301444 18.52613263 20.8550634  23.09486346]
# first 2.002-2.250, divided 47952.04795
train = np.zeros(0)
for i in range(0,len(start_split)):
    train = np.concatenate((train, orignal_train[start_split[i]:end_split[i]]))
[b, noise] = wavfile.read('noise_signal.wav')
[c, test] = wavfile.read('test_signal.wav')
noise = noise.astype(float)
"""test"""
"""ground truth"""
b_test = [105, 210, 335, 470, 570, 770, 850, 960, 1160, 1490, 1550, 1830, 1950, 2060, 2495, 2560]
e_test = [130, 230, 360, 530, 605, 830, 890, 990, 1180, 1530, 1580, 1875, 1980, 2085, 2535, 2590]
b_test, e_test = np.array(b_test), np.array(e_test)
b_test, e_test = b_test / 100 * c, e_test / 100 * c
b_test, e_test = b_test.astype(np.int), e_test.astype(np.int)
truth = np.zeros(len(test))
for i in range(len(b_test)):
    truth[b_test[i]:e_test[i]] = 1
plt.plot(truth)
plt.title("ground truth")
plt.xlabel("data points")
plt.ylabel("sound")
plt.show()
"""train_signal"""
train = train.astype(float)
desire = np.zeros(len(train))
desire[0:len(desire)-1] = train[1:len(train)]
# sampleRate = 48000
# obj = wave.open('output_train.wav','wb')
# obj.setnchannels(2)
# obj.setsampwidth(2)
# obj.setframerate(sampleRate)
# obj.writeframesraw(train)
# obj.close()
"""noise's desire"""
# desire = np.zeros(len(noise))
# desire[0:len(noise)-1] = noise[1:len(noise)]


"""LMS"""
order = 33
stepsize = 7e-11
wList, mse,mse1 = lms(train,desire,order,stepsize)
wList = np.array(wList)
wList = wList.astype(float)
w = wList[-1]
output = np.zeros(len(train))
output = output.astype(float)
for i in range(order, len(train)):
    temp = 0
    for j in range(order):
        temp = temp + w[j] * train[i-order+j]
    output[i] = temp
print("train MSE1: ",mse1[-1])
w_train = w
print("w_train: ",w_train)
"""LMS fine tuning """
# order = 24
# stepsize = 8e-11
# list = []
# for i in range(10):
#     wList, mse, mse1 = lms(train, desire, order, stepsize)
#     print("mse1 ",mse[-1]," stepsize: ",stepsize)
#     list.append(mse[-1])
#     stepsize-=1e-11
# plt.plot(list)
# plt.show()
#
# sampleRate = 48000
# obj = wave.open('output_train_LMS.wav','wb')
# obj.setnchannels(2)
# obj.setsampwidth(2)
# obj.setframerate(sampleRate)
# obj.writeframesraw(output)
# obj.close()

"""noise signal main"""
desire = np.zeros(len(noise))
desire[0:len(noise)-1] = noise[1:len(noise)]
desire[len(noise)-1] = desire[len(noise)-2]
epoch = 2
for i in range(epoch):
    desire1 = desire
    desire = np.concatenate([desire,desire1],axis = 0)
print(1)
for i in range(epoch):
    noise1 = noise
    noise = np.concatenate([noise,noise1],axis = 0)
print(2)
"""LMS"""
order = 9
stepsize = 1e-10
wList, mse, mse1 = lms(noise,desire,order,stepsize)
wList = np.array(wList)
wList = wList.astype(float)
w = wList[-1]
# output = np.zeros(len(noise))
# output = output.astype(float)
# for i in range(order, len(noise)):
#     temp = 0
#     for j in range(order):
#         temp = temp + w[j] * noise[i-order+j]
#     output[i] = temp
# print(mse[-2])
# plt.plot(mse)
# plt.show()
print("mse1: ", mse1[-1])
w_noise = w
print("w_noise: ",w_noise)
"""fine tuning"""
# order = 9
# stepsize = 1e-10
# list = []
# for i in range(10):
#     wList, mse, mse1 = lms(noise, desire, order, stepsize)
#     print("mse1 ",mse1[-1]," stepsize: ",stepsize)
#     list.append(mse1[-1])
#     stepsize-=1e-11
# plt.plot(list)
# plt.show()

"""test"""
# test = test.astype(float)
# desire = np.zeros(len(test))
# desire[0:len(desire)-1] = test[1:len(test)]
# output = np.zeros(len(test))
# output = output.astype(float)
# for i in range(len(w_train), len(test)):
#     temp = 0
#     for j in range(len(w_train)):
#         temp = temp + w_train[j] * test[i-len(w_train)+j]
#     output[i] = temp
# eList = []
# for i in range(len(w_train),len(output)):
#     e = test[i] - output[i]
#     eList.append(np.mean(e**2))
# train_error = np.asarray(eList)
# y_test_train, power_train = predict_linear_normalized(test,w_train)
# power_train = power_train[33:]
# train_error = train_error/power_train
#
# output_noise = np.zeros(len(test))
# output_noise = output.astype(float)
# for i in range(len(w_noise), len(test)):
#     temp = 0
#     for j in range(len(w_noise)):
#         temp = temp + w_noise[j] * test[i-len(w_noise)+j]
#     output_noise[i] = temp
# eList = []
# for i in range(len(w_noise),len(output_noise)):
#     e = test[i] - output_noise[i]
#     eList.append(np.mean(e**2))
# noise_error = np.asarray(eList)
# y_test_noise, power_noise = predict_linear_normalized(test,w_noise)
# power_noise = power_noise[9:]
# noise_error = noise_error/power_noise

test = test.astype(float)
desire = np.zeros(len(test))
desire[0:len(desire)-1] = test[1:len(test)]
y_test_train, power_train = predict_linear_normalized(test,w_train)
eList = []
for i in range(len(desire)):
    e = desire[i] - y_test_train[i]
    eList.append(np.mean(e**2))
train_error = np.asarray(eList)
train_error = train_error/power_train
y_test_noise, power_noise = predict_linear_normalized(test,w_noise)
eList = []
for i in range(len(desire)):
    e = desire[i] - y_test_noise[i]
    eList.append(np.mean(e**2))
noise_error = np.asarray(eList)
noise_error = noise_error/power_noise

"""put average error here"""
error_train = average(train_error,8200)
error_noise = average(noise_error,8200)
#diff = 24 # train order 33, noise order 9, start from 0,diff = 24+1= 25
#error_noise = error_noise[diff:]
y = np.zeros(len(error_noise))
for i in range (len(error_train)):
    if(error_train[i] < error_noise[i]):
        y[i] = 1
    else:
        y[i] = 0
plt.plot(y)
plt.title("predicted manatee calls")
plt.xlabel("data points")
plt.ylabel("manatee call")
plt.show()




"""predict"""
# y = np.zeros(len(test))
# y = y.astype(float)
# def wienerTakeAll(y, ind, range1, cutoff,maxlength):
#     train = 1
#     noise = 1
#     for i in range (ind-range1,ind+range1):
#         if(i < 0):
#             i = 0
#         if(i >= maxlength):
#             break
#         if (train_error[i] < noise_error[i]):
#             train+=1
#         else:
#             noise+=1
#     if(train/noise > cutoff):
#         y[ind] = 1
#     pass
#
# for i in range (len(w_noise),min(len(noise_error),len(train_error))):
#     wienerTakeAll(y,i,250,1.2,min(len(noise_error),len(train_error)))
# plt.plot(y)
# plt.show()

y_smooth1 = np.zeros(len(y))
y_smooth1[0:len(y_smooth1)] = y[0:len(y_smooth1)]
# plt.plot(y_smooth1)
# plt.show()
# def deepSmooth(y, smooth1):
#     for i in range(smooth1,len(y)-smooth1):
#         if(y[i] == 1):
#             y[i:i+smooth1] = 1
#             y[i + smooth1: 2*(i+smooth1)] = 0
#             i += smooth1
# deepSmooth(y_smooth1,2)
# plt.plot(y_smooth1)
# plt.show()
# for i in range(max(len(w_noise),len(w_train)),min(len(noise_error),len(train_error))):
#     if(train_error[i] < noise_error[i]):
#         y[i] = 1


#r = average(y, 500, 0.58)
# r = average(y_smooth1, 10000, 0.01)
# plt.plot(r)
# plt.title("Manatee call after smooth")
# plt.xlabel("data points")
# plt.ylabel("manatee call")
# plt.show()

# timeDomain = np.zeros(len(r))
# for i in range(len(r)):
#     timeDomain[i] = i/441
# plt.plot(timeDomain,r)
# plt.title("Manatee call after smooth")
# plt.xlabel("time(ms)")
# plt.ylabel("manatee call")
# plt.show()

# audio = np.zeros(len(test))
# for i in range(len(test)-len(r),len(r)):
#     if(r[i] == 1):
#         audio[i] = output[i]
#     else:
#         audio[i] = output_noise[i]
# sampleRate = 48000
# obj = wave.open('output_test_LMS.wav','wb')
# obj.setnchannels(2)
# obj.setsampwidth(1)
# obj.setframerate(sampleRate)
# obj.writeframesraw(audio)
# obj.close()

"""part B"""
w_manatee = wiener(train[:-1], train[1:], 200)
w_noise = wiener(noise[:-1], noise[1:], 9)
'''get statistics'''
y_t = predict_linear(train[:-1], w_train)
y_n = predict_linear(noise[:-1], w_noise)
e_t = train[1:] - y_t
e_n = noise[1:] - y_n
mean_m, mean_n = np.mean(e_t), np.mean(e_n)
std_m, std_n = np.std(e_t), np.std(e_n)

y_t = predict_linear(test[:-1], w_manatee)
y_n = predict_linear(test[:-1], w_noise)
e_t = test[1:] - y_t
e_n = test[1:] - y_n

b_test = [105, 210, 335, 470, 570, 770, 850, 960, 1160, 1490, 1550, 1830, 1950, 2060, 2495, 2560]
e_test = [130, 230, 360, 530, 605, 830, 890, 990, 1180, 1530, 1580, 1875, 1980, 2085, 2535, 2590]
b_test, e_test = np.array(b_test), np.array(e_test)
b_test, e_test = b_test / 100 * c, e_test / 100 * c
b_test, e_test = b_test.astype(np.int), e_test.astype(np.int)
mask = np.zeros(test.size)
for i in range(b_test.size):
    mask[b_test[i]:e_test[i]] = 1

y_t = predict_linear(test[:-1], w_manatee)
y_n = predict_linear(test[:-1], w_noise)
e_t = test[1:] - y_t
e_n = test[1:] - y_n
'''SPRT'''
s_m = e_t[:100]
s_n = e_n[:100]
p_m = gaussian(s_m, mean_m, std_m)
p_n = gaussian(s_n, mean_n, std_n)
d = np.sum(np.log(p_m)) - np.sum(np.log(p_n))  # 看 d 决定最开始是哪一个
print(d)
'''CUSUM'''
p_m = gaussian(e_t, mean_m, std_m)
p_n = gaussian(e_n, mean_n, std_n)
l_10 = np.log(p_m / p_n)
l_01 = np.log(p_n / p_m)
label = np.zeros(test.size - 1)
label[:100] = 0
threshold = 25000
delta_L = np.zeros(test.size - 1)
tem_l = 0
for i in range(100, delta_L.size):
    if label[i - 1] == 0:
        tem_l = l_10[i]
    elif label[i - 1] == 1:
        tem_l = l_01[i]
    delta_L[i] = max(0, delta_L[i - 1] + tem_l)
    if delta_L[i] > threshold:
        label[i] = 1 - label[i - 1]
        delta_L[i] = 0

plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(mask)
plt.title('truth')
plt.xlabel('time(s)')
#plt.subplot(212)
plt.show()
plt.plot(label)
plt.title('CUSUM predicted calls')
plt.xlabel('data points')
plt.ylabel('manatee call')
plt.show()

"""roc curve"""
rocy = [0,0.2,0.4,0.6,0.8,1]
rocx = [0,0.07, 0.143,  0.143, 0.143,1]

procy = [0,0.2,0.4,0.6,0.8,1]
procx = [0,0,0,0,0,1]
plt.plot(rocx,rocy, label = "linear model")
plt.plot(procx, procy, label = "CUSUM model")
plt.title("ROC curve")
plt.xlabel("false positive")
plt.ylabel("sensitivity")
plt.legend()
plt.show()