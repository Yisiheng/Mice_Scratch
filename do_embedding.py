# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:20:54 2022

@author: h'h
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import statsmodels.tsa.api as sm
import gudhi
from gudhi.point_cloud import timedelay
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import csv
import scipy
import os
import numpy.linalg as linalg
from scipy import sparse




#Str='E:\\小鼠\\negative\\negative\\202206021135-1_02-167-51550-51685.mkv'
#Str='E:\\小鼠\\positive\\positive\\202206020911-0_04-30-10002-10226.mkv'


def Observe(Data):
    #observe the data about video (swith the video to time series)
    #input:
        #Data: (np.array) the flow of image
    #return:
        #time_series: (np.array) 
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
    #calculate the appropriate the dimension of embedding
    #input:
        #Sq: (np.array of list of number) time series
    #return:
        #dim: the dimension of embedding
    
    dim=3
    return dim
    

def Choose_delay(Sq, Dim = 3):
    #choose a appropriate delay parameter
    #input:
        #Sq: (np.array or list of number) time series
        #Dim: the dimension of embedding
    #return:
        #delay: (int) the parameter, delay
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


def Delay_embed(Data, Delay, Dim = 3, Skip = 1):
    
    point_Cloud = timedelay.TimeDelayEmbedding(dim = Dim, delay = Delay, skip = Skip)
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



def getSlidingWindowVideo(I, dim, Tau, dT):
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



def getPCAVideo(I):
    ICov = I.dot(I.T)
    [lam, V] = linalg.eigh(ICov)
    lam[lam < 0] = 0
    V = V*np.sqrt(lam[None, :])
    return V




def estimateFundamentalFreq(x, doPlot = False):
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


def fundamentalFreqEstimation(X):
    #Do Diffusion Maps
    DOrig = getSSM(X)
    XDiffused = getDiffusionMap(DOrig, 0.1)
    x = XDiffused[:, -2] #Get the mode corresponding to the largest eigenvalue
    x = x - np.mean(x)
    (maxT, corr) = estimateFundamentalFreq(x)
    return (x, maxT, corr)




def getDiffusionMap(SSM, Kappa, t = -1, includeDiag = True, thresh = 5e-4, NEigs = 10):
    """
    :param SSM: Metric between all pairs of points
    :param Kappa: Number in (0, 1) indicating a fraction of nearest neighbors
                used to autotune neighborhood size
    :param t: Diffusion parameter.  If -1, do Autotuning
    :param includeDiag: If true, include recurrence to diagonal in the markov
        chain.  If false, zero out diagonal
    :param thresh: Threshold below which to zero out entries in markov chain in
        the sparse approximation
    :param NEigs: The number of eigenvectors to use in the approximation
    """
    N = SSM.shape[0]
    #Use the letters from the delaPorte paper
    K = getW(SSM, int(Kappa*N))
    if not includeDiag:
        np.fill_diagonal(K, np.zeros(N))
    RowSumSqrt = np.sqrt(np.sum(K, 1))
    DInvSqrt = sparse.diags([1/RowSumSqrt], [0])

    #Symmetric normalized Laplacian
    Pp = (K/RowSumSqrt[None, :])/RowSumSqrt[:, None]
    Pp[Pp < thresh] = 0
    Pp = sparse.csr_matrix(Pp)

    lam, X = sparse.linalg.eigsh(Pp, NEigs, which='LM')
    lam = lam/lam[-1] #In case of numerical instability

    #Check to see if autotuning
    if t > -1:
        lamt = lam**t
    else:
        #Autotuning diffusion time
        lamt = np.array(lam)
        lamt[0:-1] = lam[0:-1]/(1-lam[0:-1])

    #Do eigenvector version
    V = DInvSqrt.dot(X) #Right eigenvectors
    M = V*lamt[None, :]
    return M/RowSumSqrt[:, None] #Put back into orthogonal Euclidean coordinates


def getSSM(X):
    XSqr = np.sum(X**2, 1)
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D = np.sqrt(D)
    return D



def getW(D, K, Mu = 0.5):
    """
    Return affinity matrix
    [1] Wang, Bo, et al. "Similarity network fusion for aggregating data types on a genomic scale." 
        Nature methods 11.3 (2014): 333-337.
    :param D: Self-similarity matrix
    :param K: Number of nearest neighbors
    """
    #W(i, j) = exp(-Dij^2/(mu*epsij))
    DSym = 0.5*(D + D.T)
    np.fill_diagonal(DSym, 0)

    Neighbs = np.partition(DSym, K+1, 1)[:, 0:K+1]
    MeanDist = np.mean(Neighbs, 1)*float(K+1)/float(K) #Need this scaling
    #to exclude diagonal element in mean
    #Equation 1 in SNF paper [1] for estimating local neighborhood radii
    #by looking at k nearest neighbors, not including point itself
    Eps = MeanDist[:, None] + MeanDist[None, :] + DSym
    Eps = Eps/3
    W = np.exp(-DSym**2/(2*(Mu*Eps)**2))
    return W


def getSSM(X):
    #距离矩阵
    XSqr = np.sum(X**2, 1)
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D[D < 0] = 0 #Numerical precision
    D = np.sqrt(D)
    return D


def getPCAVideo(I):
    ICov = I.dot(I.T)
    [lam, V] = linalg.eigh(ICov)
    lam[lam < 0] = 0
    V = V*np.sqrt(lam[None, :])
    return V













