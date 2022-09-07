# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 19:07:52 2022

@author: h'h
"""

import numpy as np
import cv2
import os

def Read_video(Path_file):
    #load video file and return np.array :shape=(num, length, width)
    
    #input:
        #Path_file: the path of video file, type is string
    #return:
        #np.array (:shape=(num, length, width))
    data = []
    cap = cv2.VideoCapture(Path_file)
    
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
        
        k = cv2.waitKey(20)
        #按q键退
        if(k & 0xff == ord('q')):
            break
    
    cap.release()
    cv2.destroyAllWindows
    
    return np.array(data)


def Difference_frame(Data):
    #do temporal difference for the data of video
    
    #input:
        #Data: video data, np.array
    #return: 
        #np.array (the data by difference)
    #做帧间差分
    data=[]
    
    #降噪并做差分帧
    for i in range(len(Data) - 1):
        gray_frame_front = cv2.GaussianBlur(Data[i],(3,3),0)
        gray_frame_later = cv2.GaussianBlur(Data[i+1],(3,3),0)
        d_frame = cv2.absdiff(gray_frame_front, gray_frame_later)
        ret, d_frame = cv2.threshold(d_frame, 5,255, cv2.THRESH_BINARY)
        data.append(d_frame)

    return np.array(data)


def Out_differencing_video(Path_file):
    #do difference frame, and out to path which is assigned.
    
    #input:
        #Path_file: address of video file 
    
    #Fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #Fps=30
    #Size = (640, 480) = (frame[1], frame[0])
    #Str='E:\\mice\\out.avi'
    
    cap = cv2.VideoCapture(Path_file)
    # 设置需要保存视频的格式“xvid”
    # 该参数是MPEG-4编码类型，文件名后缀为.avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 设置视频帧频
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 设置视频大小
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 释放资源
    cap.release()
    
    data=Read_video(Path_file)
    diff_data=Difference_frame(data)
    
    # VideoWriter方法是cv2库提供的保存视频方法
    # 按照设置的格式来out输出
    out = cv2.VideoWriter('E:\\mice\\out.avi', fourcc, fps, size, 0)
    
    for frame in diff_data:
        #frame = cv2.flip(frame,0)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'): break
    # 释放资源
    out.release()
    
    # 关闭窗口
    cv2.destroyAllWindows()
    

def Walk_file(Path_file):
    #遍历整个文件夹下的所有文件，返回所有文件路径的list
    #input:
        #Path_file: Traversed folder address, string
    #return:
        #list of aadress of files in Path_file
    Paths=[]
    for root, dirs, files in os.walk(Path_file):
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
