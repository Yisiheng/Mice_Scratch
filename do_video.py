# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:20:53 2022

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


#Str='E:\\小鼠\\negative\\negative\\202206021135-1_02-167-51550-51685.mkv'
#Str='E:\\小鼠\\positive\\positive\\202206020911-0_04-30-10002-10226.mkv'



def Walk_file(file):
    #through all files in the specified folder and return a list of all file paths
    #input:
        #Path: (string) the address of specified folder
    #return:
        #Paths: (list of string) all file paths in specified folder
    
    
    Paths=[]
    for root, dirs, files in os.walk(file):
    # root 表示当前正在访问的文件夹路径
    # dirs 表示该文件夹下的子目录名list
    # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            Paths.append(os.path.join(root, f))
            #print(os.path.join(root, f))
        '''
        # 遍历所有的文件夹
        for d in dirs:
            print('b')
            print(os.path.join(root, d))
        '''
    return Paths




def Read_csv(Str):
    
    sq=[]
    with open(Str, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)
            sq.append(row)



def Read_video(Path, Key = 0, Show_video=False):
    #read the video
    #input: 
        #Path: (string) the address of the video
        #Key: (0 or 1) indicates whether to change the image into a vector
    #return:
        #data: (np.array) the flow of image, the size is (NFrames, size)
    
    if not os.path.exists(Path):
       print("ERROR: Video path not found: %s"%Path)
       return None    
    
    cap = cv2.VideoCapture(Path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps is ', fps)
    
    NFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    
    if Key == 0:
        data = np.zeros((NFrames, size[0], size[1]))
    
        i = 0
        while(cap.isOpened()):
            ret, imgRGB = cap.read()
            if not ret :
                break
            imGray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
            
            data[i,:,:] = imGray
            i = i+1
            if Show_video:
                cv2.imshow('image',imgRGB)
            k = cv2.waitKey(20)
            #按q键退
            if(k & 0xff == ord('q')):
                break   

        cap.release()
        cv2.destroyAllWindows
    else:
        data = np.zeros((NFrames,size[0]*size[1]))
    
        i = 0
        while(cap.isOpened()):
            ret, imgRGB = cap.read()
            if not ret :
                break
            imGray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
            
            data[i,:] = imGray.flatten()
            i = i+1
            if Show_video:
                cv2.imshow('image',imgRGB)
            k = cv2.waitKey(20)
            #按q键退
            if(k & 0xff == ord('q')):
                break   

        cap.release()
        cv2.destroyAllWindows
    
    return (data,size)



def Difference_frame(Data):
    #do difference frame
    #input:
        #Data: (np.array) the flow of image
    #return:
        #(np.array) the flow of difference_frame
    
    data=[]
    
    #降噪并做差分帧
    for i in range(len(Data) - 1):
        gray_frame_front = cv2.GaussianBlur(Data[i],(3,3),0)
        gray_frame_later = cv2.GaussianBlur(Data[i+1],(3,3),0)
        d_frame = cv2.absdiff(gray_frame_front, gray_frame_later)
        ret, d_frame = cv2.threshold(d_frame, 5,255, cv2.THRESH_BINARY)
        data.append(d_frame)

    return np.array(data)

def Produce_video(Sq,Path='E:\\mice\\out.avi'):
    #swith the flow of image to video
    #input:    
        #Sq: (np.array) time series of image
        #Path: (string)
    #return: None
    
    #Fps=30
    #Size = (640, 480)
    #Fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # 设置需要保存视频的格式“xvid”
    # 该参数是MPEG-4编码类型，文件名后缀为.avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 设置视频帧频
    fps = 30
    # 设置视频大小
    size = (640,480)

    # VideoWriter方法是cv2库提供的保存视频方法
    # 按照设置的格式来out输出
    out = cv2.VideoWriter(Path, fourcc, fps, size, 0)
    
    for frame in Sq:
        #frame = cv2.flip(frame,0)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'): break
    # 释放资源
    out.release()
    
    # 关闭窗口
    cv2.destroyAllWindows()
    
    
#针对单一图片进行降维，采取卷积的方法
def Filter_weight(image,weight,skip):
    
    plt.figure("image")
    plt.imshow(image)
    
    height, width = image.shape
    h, w = weight.shape
    sh, sw = skip[0], skip[1]
    new_h,new_w = int((height - h)/sh) + 1, int((width - h)/sw) +1
    new_image = np.zeros((new_h, new_w), dtype=np.float)
    
    #打马赛克
    New_image = np.zeros((new_h, new_w), dtype=np.float)
    
    for i in range(new_h):
        for j in range(new_w):
            
            val=np.sum(image[i*sh:i*sh+h,j*sw:j*sw+w]*weight)
            val=int(val/(w*h))
            new_image[i,j]=val
            New_image[i*sh:i*sh+h,j*sw:j*sw+w]=np.ones((h,w))*val
            
    
    #cv2.imshow('img',new_image)
    plt.figure("new_image")
    plt.imshow(new_image)
    
    return new_image

#针对单一图片进行打码操作
def Mosaic(Image, Msize):
    
    plt.figure("image")
    plt.imshow(Image)
    
    height, width = Image.shape
    h, w = Msize[0], Msize[1]
    H, W = int(height/h), int(width/w)
    
    New_image=np.ones((height,width), dtype=np.float)
    weight=np.ones((h,w))
    
    for i in range(H):
        for j in range(W):
            
            val=np.sum(Image[i*h:(i+1)*h,j*w:(j+1)*w]*weight)
            val=int(val/(w*h))
            New_image[i*h:(i+1)*h,j*w:(j+1)*w]=np.ones((h,w))*val
            
    
    #cv2.imshow('img',new_image)
    plt.figure("New_image")
    plt.imshow(New_image)
    
    #New_image_1 = New_image.astype("uint8")
    
    return New_image.astype('uint8')



#分离前景与背景其中有加入去噪的操作
def Split_BG(Image):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.createBackgroundSubtractorKNN()

    fgmask = fgbg.apply(Image)
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_ELLIPSE,kernel)
        
    cv2.imshow('frame',fgmask)



#降噪
def reduce_Noise():
    1
    

def Get_PCA_video(Path, Score = 0.95):
    
    XOrig,shape=Read_video(Path, Key = 1)
    
    pca = PCA(Score)
    reduced_data = pca.fit_transform(XOrig)
    data_pca=pca.inverse_transform(reduced_data)

    # VideoWriter方法是cv2库提供的保存视频方法
    
    # 设置需要保存视频的格式“xvid”
    # 该参数是MPEG-4编码类型，文件名后缀为.avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 设置视频帧频
    fps = 30
    # 设置视频大小
    size = (640,480)
    # 按照设置的格式来out输出
    out = cv2.VideoWriter('E:\\mice\\out.avi', fourcc, fps, size, 0)

    for frame in data_pca:
        frame=frame.reshape(480,640)
        frame=frame.astype('uint8')
        #frame = cv2.flip(frame,0)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'): break
    
    # 释放资源
    out.release()

        # 关闭窗口
    cv2.destroyAllWindows()


def Get_Mosaic_Video(Path, Msize = (20,20)):
    
    XOrig,shape=Read_video(Path, Key = 0)
    
    Nframe, length, width = XOrig.shape
    
    # 设置需要保存视频的格式“xvid”
    # 该参数是MPEG-4编码类型，文件名后缀为.avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 设置视频帧频
    fps = 30
    # 设置视频大小
    size = (640,480)
    # 按照设置的格式来out输出
    out = cv2.VideoWriter('E:\\mice\\out.avi', fourcc, fps, size, 0)

    for frame in XOrig:
        frame=Mosaic(frame, Msize)
        #frame = cv2.flip(frame,0)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'): break
    
    # 释放资源
    out.release()

        # 关闭窗口
    cv2.destroyAllWindows()




