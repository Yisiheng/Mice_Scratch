# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 19:07:53 2022

@author: h'h
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as sm
from gudhi.point_cloud import timedelay
from sklearn.decomposition import PCA
import scipy
from mpl_toolkits.mplot3d import Axes3D

def Observe(Data):
    #input:
        #Data: raw data in some unknow manifold
    #return:
        #time series
    
    M , N = Data[0].shape
    
    time_series = []
    for x in Data:
        value=0
        for i in range(M):
            for j in range(N):
                value=value+x[i,j]
        time_series.append(value)
    time_series = np.array(time_series)
    time_series = time_series-np.mean(time_series)
    time_series = time_series/np.max(time_series)
    
    return time_series

def Determine_dimension(Sq):
    #从时间序列中计算出合适的嵌入维数
    1
    

def Choose_delay(Sq, Dim = 3):
    #choose appropriate parameter, delay
    #input:
        #Sq: time series, list of number
        #Dim: the dimension of embedding
    #return:
        #point cloud, np.array
    
    Acf=sm.stattools.acf(Sq, nlags = int((len(Sq) + 1)/2))
    
    #找到最合适的参数,及自相关系数中的最大的峰值,maxTau
    zc = np.zeros(Acf.size-1)
    zc[(Acf[0:-1] < 0)*(Acf[1::] > 0)] = 1
    zc[(Acf[0:-1] > 0)*(Acf[1::] < 0)] = -1

    #Mark regions which are admissible for key maxes
    #(regions with positive zero crossing to left and negative to right)
    admiss = np.zeros(Acf.size)
    admiss[0:-1] = zc
    for i in range(1, Acf.size):
        if admiss[i] == 0:
            admiss[i] = admiss[i-1]

    #Find all local maxes
    maxes = np.zeros(Acf.size)
    maxes[1:-1] = (np.sign(Acf[1:-1] - Acf[0:-2])==1)*(np.sign(Acf[1:-1] - Acf[2::])==1)
    maxidx = np.arange(Acf.size)
    maxidx = maxidx[maxes == 1]
    maxTau = 0
    if len(Acf[maxidx]) > 0:
        maxTau = maxidx[np.argmax(Acf[maxidx])]
    #Acf_sorted=sorted(Acf)
    '''
    delay = np.where(Acf == np.max(Acf[1:]))[0][0]/Dim
    
    if delay - int(delay) > 0.5:
        delay = int(delay) + 1
    else:
        delay = int(delay)
    
    if delay == 0:
        delay = 1
    '''
    
    delay=maxTau/Dim
    delay=int(delay+0.5)
    
    return max(1,delay)
    
def estimateFundamentalFreq(x, doPlot = False):
    #别人的代码!
    #Step 1: Compute normalized squared difference function
    #Using variable names in the paper
    N = x.size
    W = np.int(N/2)
    t = W
    corr = np.zeros(W)
    #Do brute force f FFT because I'm lazy
    #(fine because signals are small)
    for Tau in np.arange(W):
        xdelay = x[Tau::]
        L = (W - Tau)/2
        m = np.sum(x[int(t-L):int(t+L+1)]**2) + np.sum(xdelay[int(t-L):int(t+L+1)]**2)
        r = np.sum(x[int(t-L):int(t+L+1)]*xdelay[int(t-L):int(t+L+1)])
        corr[Tau] = 2*r/m

    #Step 2: Find the ''key max''
    #Compute zero crossings
    zc = np.zeros(corr.size-1)
    zc[(corr[0:-1] < 0)*(corr[1::] > 0)] = 1
    zc[(corr[0:-1] > 0)*(corr[1::] < 0)] = -1

    #Mark regions which are admissible for key maxes
    #(regions with positive zero crossing to left and negative to right)
    admiss = np.zeros(corr.size)
    admiss[0:-1] = zc
    for i in range(1, corr.size):
        if admiss[i] == 0:
            admiss[i] = admiss[i-1]

    #Find all local maxes
    maxes = np.zeros(corr.size)
    maxes[1:-1] = (np.sign(corr[1:-1] - corr[0:-2])==1)*(np.sign(corr[1:-1] - corr[2::])==1)
    maxidx = np.arange(corr.size)
    maxidx = maxidx[maxes == 1]
    maxTau = 0
    if len(corr[maxidx]) > 0:
        maxTau = maxidx[np.argmax(corr[maxidx])]

    if doPlot:
        plt.subplot(211)
        plt.plot(x)
        plt.title("Original Signal")
        plt.subplot(212)
        plt.plot(corr)
        plt.hold(True)
        plt.plot(admiss*1.05, 'r')
        plt.ylim([-1.1, 1.1])
        plt.scatter(maxidx, corr[maxidx])
        plt.scatter([maxTau], [corr[maxTau]], 100, 'r')
        plt.title("Max Tau = %i, Clarity = %g"%(maxTau, corr[maxTau]))
    return (maxTau, corr)

def getSlidingWindowVideo(I, dim, Tau, dT):
    #别人的代码
    N = I.shape[0] #Number of frames
    P = I.shape[1] #Number of pixels (possibly after PCA)
    pix = np.arange(P)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim*P))
    idx = np.arange(N)
    for i in range(NWindows):
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))
        f = scipy.interpolate.interp2d(pix, idx[start:end+1], I[idx[start:end+1], :], kind='linear')
        X[i, :] = f(pix, idxx).flatten()
    return X

def Delay_embedding(Data, Delay, Dim = 3):
    #slding window embedding
    
    point_Cloud = timedelay.TimeDelayEmbedding(dim = Dim, delay = Delay, skip = 1)
    Points = point_Cloud(Data)
    
    return Points



def Use_PCA(Data, Dim=3):
    #dimension reduction using PCA
    
    #input:
        #Data: np.array
        #Dim: Number of feature dimensions after dimension reduction, int
    #return:
        #np.array, the data after dimension reduction
    
    pca = PCA(n_components = Dim)
    reduced_data = pca.fit_transform(Data)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2])
    ax.plot3D(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2], 'gray')

    return reduced_data























