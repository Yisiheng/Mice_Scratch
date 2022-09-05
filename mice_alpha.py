# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 09:32:40 2022

@author: h'h
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab
import imageio
import skimage.io
import cv2
import statsmodels.tsa.api as sm
import gudhi
from gudhi.point_cloud import timedelay
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import csv
import numpy.linalg as linalg
import scipy


def Read_csv(Str):
    
    sq=[]
    with open(Str, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)
            sq.append(row)

def Read_video(Str):
    
    data = []
    cap = cv2.VideoCapture(Str)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps is ', fps)
    
    while(cap.isOpened()):
        ret,imgRGB = cap.read()
        if ret:
            
            imGray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
            data.append(imGray)
        
            #cv2.imshow('image',imgRGB)
        else:
            break
        '''
        k = cv2.waitKey(20)
        #按q键退
        if(k & 0xff == ord('q')):
            break
    '''
    cap.release()
    cv2.destroyAllWindows
    
    return np.array(data)


def Difference_frame(Data):
    
    data=[]
    
    #降噪并做差分帧
    for i in range(len(Data) - 1):
        gray_frame_front = cv2.GaussianBlur(Data[i],(3,3),0)
        gray_frame_later = cv2.GaussianBlur(Data[i+1],(3,3),0)
        d_frame = cv2.absdiff(gray_frame_front, gray_frame_later)
        ret, d_frame = cv2.threshold(d_frame, 5,255, cv2.THRESH_BINARY)
        data.append(d_frame)

    return np.array(data)


def Out_differencing_video(Str):
    
    #Fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #Fps=30
    #Size = (640, 480) = (frame[1], frame[0])
    #Str='E:\\mice\\out.avi'
    
    cap = cv2.VideoCapture(Str)
    # 设置需要保存视频的格式“xvid”
    # 该参数是MPEG-4编码类型，文件名后缀为.avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    
    data=Read_video(Str)
    diff_data=Difference_frame(data)
    
    out = cv2.VideoWriter('E:\\mice\\out.avi', fourcc, fps, size, 0)
    for frame in diff_data:
        #frame = cv2.flip(frame,0)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'): break
    out.release()
    
    cv2.destroyAllWindows()
    
'''
    cap = cv2.VideoCapture('E:\\小鼠\\positive\\positive\\202206020911-0_01-142-53674-53700.mkv')
    # 设置需要保存视频的格式“xvid”
    # 该参数是MPEG-4编码类型，文件名后缀为.avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 设置视频帧频
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 设置视频大小
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # VideoWriter方法是cv2库提供的保存视频方法
    # 按照设置的格式来out输出
    out = cv2.VideoWriter('E:\\mice\\out.avi',fourcc ,fps, size, 0)

    # 确定视频打开并循环读取
    while(cap.isOpened()):
        # 逐帧读取，ret返回布尔值
        # 参数ret为True 或者False,代表有没有读取到图片
        # frame表示截取到一帧的图片
        ret, frame = cap.read()
        if ret == True:
            # 垂直翻转矩阵
            frame = cv2.flip(frame,0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # 释放资源
    cap.release()
    out.release()
    # 关闭窗口
    cv2.destroyAllWindows()
'''
    

def Observe(Data):
    
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
    1
    

def Choose_delay(Sq, Dim = 3):
    
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

def Delay_embed(Data, Delay, Dim = 3):
    
    point_Cloud = timedelay.TimeDelayEmbedding(dim = Dim, delay = Delay, skip = 1)
    Points = point_Cloud(Data)
    
    return Points


def Persistence(Points, Max_edge_length = 2.0, Max_dimension = 2):
    
    rips_complex=gudhi.RipsComplex(points = Points,max_edge_length = Max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension = Max_dimension)
    
    diag = simplex_tree.persistence(homology_coeff_field = 2, min_persistence = 0.1, persistence_dim_max = False)
    
    return diag


def Split_diag(Diag):
    
    diag_0 = []
    diag_1 = []
    diag_2 = []
    for x in Diag:
        if x[0] == 0: diag_0.append(x)
        elif x[0] == 1: diag_1.append(x)
        else: diag_2.append(x)
    return diag_0, diag_1, diag_2
    

def Describe_persistence(Diag):
    
    gudhi.plot_persistence_barcode(persistence=Diag, max_intervals=100, inf_delta=0.1, legend=True)
    gudhi.plot_persistence_diagram(persistence=Diag, max_intervals=100, inf_delta=0.1, legend=True, greyblock=True)


def Use_PCA(Data, Dim=3):
    
    pca = PCA(n_components = Dim)
    reduced_data = pca.fit_transform(Data)
    
    return reduced_data



#Str='E:\\小鼠\\positive\\positive\\202206020911-0_01-142-53674-53700.mkv'
#Str='E:\\小鼠\\negative\\negative\\202206021135-1_02-167-51550-51685.mkv'
#Str='E:\\小鼠\\20220608\\202206081230-0_00.mkv'
def f_1(Str):
    #非差分帧，直接对每张图片求和
    
    data = Read_video(Str)
    sq = Observe(data)
    
    #dim = Determine_dimension(sq)
    dim=10
    
    delay = Choose_delay(sq, dim)
    
    
    points = Delay_embed(sq, delay, Dim = dim)
    
    #drawing
    plt.figure()
    plt.plot(sq)
    
    
    reduced_data = Use_PCA(points)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2])
    ax.plot3D(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2], 'gray')
    
    diag = Persistence(points, Max_edge_length = 10.0, Max_dimension = 3)
    Describe_persistence(diag)


    
def f_2(Str):
    #做差分帧，再对每帧求和
    
    data = Read_video(Str)
    
    diff_data=Difference_frame(data)
    
    sq = Observe(diff_data)
    
    #dim = Determine_dimension(sq)
    dim = 10
    
    delay = Choose_delay(sq, dim)
    
    points = Delay_embed(sq, delay, Dim = dim)
    
    #drawing
    plt.figure()
    plt.plot(sq)
    
    reduced_data = Use_PCA(points)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2])
    ax.plot3D(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2], 'gray')

    diag = Persistence(points, Max_edge_length = 2.0, Max_dimension = 3)
    Describe_persistence(diag)


def f_3(Str):
    #对长视频进行操作，内存爆炸，待优化！！！！！！！！！！！
    
    data = []
    cap = cv2.VideoCapture(Str)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps is ', fps)
    
    while(cap.isOpened()):
        ret,imgRGB = cap.read()
        if ret:
            
            imGray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
            data.append(imGray)
        
            #cv2.imshow('image',imgRGB)
        else:
            break
        '''
        k = cv2.waitKey(20)
        #按q键退
        if(k & 0xff == ord('q')):
            break
    '''
    cap.release()
    cv2.destroyAllWindows
    
    data = Read_video(Str)
    
    
    data = data[1:] - data[:-1]
    
    sq = Observe(data)
    
    dim = 20
    
    delay = Choose_delay(sq, dim)
    points = Delay_embed(sq, delay, Dim = dim)

    plt.figure()
    plt.plot(sq)

    reduced_data = Use_PCA(points)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2])    

    diag = Persistence(points, Max_edge_length = 2.0, Max_dimension = 3)
    Describe_persistence(diag)






