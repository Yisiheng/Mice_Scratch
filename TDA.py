# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 21:36:54 2022

@author: h'h
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gudhi
import cv2
import os
from proceed_video import *
from reconstruct_geometry import *



def Persistence(Points, Max_edge_length = 2.0, Max_dimension = 2):
    
    rips_complex=gudhi.RipsComplex(points = Points,max_edge_length = Max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension = Max_dimension)
    
    diag = simplex_tree.persistence(homology_coeff_field = 2, min_persistence = 0.1, persistence_dim_max = False)
    
    return diag


def Split_diag(Diag):
    #split diag according to dimensions, the default splitting is 0-D, 1-D and 2-D. 
    
    diag_0 = []
    diag_1 = []
    diag_2 = []
    for x in Diag:
        if x[0] == 0: diag_0.append(x)
        elif x[0] == 1: diag_1.append(x)
        else: diag_2.append(x)
    return diag_0, diag_1, diag_2



#Str='E:\\小鼠\\positive\\positive\\202206020911-0_01-142-53674-53700.mkv'
#Str='E:\\小鼠\\negative\\negative\\202206021135-1_02-167-51550-51685.mkv'
#Str='E:\\小鼠\\20220608\\202206081230-0_00.mkv'

def f_1(Str):
    #处理单个视频，非帧间差分，直接对每张图片求和
    
    data = Read_video(Str)
    sq = Observe(data)
    
    #dim = Determine_dimension(sq)
    dim=10
    
    delay = Choose_delay(sq, dim)
    
    
    points = Delay_embedding(sq, delay, Dim = dim)
    
    #drawing
    plt.figure()
    plt.plot(sq)
    
    reduced_data = Use_PCA(points)
    
    diag = Persistence(points, Max_edge_length = 10.0, Max_dimension = 3)
    gudhi.plot_persistence_barcode(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True)
    gudhi.plot_persistence_diagram(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True, greyblock=True)


    
def f_2(Str):
    #处理单个视频，做差分帧，再对每帧求和
    
    data = Read_video(Str)
    
    diff_data = Difference_frame(data)
    
    sq = Observe(diff_data)
    
    #dim = Determine_dimension(sq)
    dim = 10
    
    delay = Choose_delay(sq, dim)
    
    points = Delay_embedding(sq, delay, Dim = dim)
    
    #drawing
    plt.figure()
    plt.plot(sq)
    
    reduced_data = Use_PCA(points)


    diag = Persistence(points, Max_edge_length = 2.0, Max_dimension = 3)
    gudhi.plot_persistence_barcode(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True)
    gudhi.plot_persistence_diagram(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True, greyblock=True)


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
    points = Delay_embedding(sq, delay, Dim = dim)

    plt.figure()
    plt.plot(sq)

    reduced_data = Use_PCA(points)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2])    

    diag = Persistence(points, Max_edge_length = 2.0, Max_dimension = 3)
    gudhi.plot_persistence_barcode(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True)
    gudhi.plot_persistence_diagram(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True, greyblock=True)



def f_4(File_path):
    #文件夹路径，批量处理该文件夹下的所有文件，并保存(非帧间差分)
    
    Paths = Walk_file(File_path)
    
    for path in Paths:
        
        data = Read_video(path)
        sq = Observe(data)
        
        path, _ = os.path.splitext(path)
        
        #判断path文件夹是否存在，如果不存在则创建path文件夹
        if os.path.exists(path):
            continue
        else:
            os.mkdir(path)
        
        #生成波形图并保存
        plt.figure()
        plt.plot(sq)
        plt.savefig(path+'\\oscillogram.png')
        plt.close()
        
        #dim = Determine_dimension(sq)
        dim=10
        
        delay = Choose_delay(sq, dim)
        
        #做滑窗嵌入
        points = Delay_embedding(sq, delay, Dim = dim)
        
        #PCA后生成可视图像
        reduced_data = Use_PCA(points)
        plt.savefig(path+'\\3D.png')
        plt.close()
        
        diag = Persistence(points, Max_edge_length = 10.0, Max_dimension = 3)
        gudhi.plot_persistence_barcode(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True)
        plt.savefig(path+'\\barcode.png')
        plt.close()
        gudhi.plot_persistence_diagram(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True, greyblock=True)
        plt.savefig(path+'\\diagram.png')
        plt.close()
        
        plt.close('all')


def f_5(File_path):
    #文件夹路径，批量处理该文件夹下的所有文件，并保存(帧间差分)
    
    Paths = Walk_file(File_path)
    
    for path in Paths:
        
        data = Read_video(path)
        diff_data = Difference_frame(data)
        sq = Observe(diff_data)
        
        path, _ = os.path.splitext(path)
        path=path+'_diff'
        
        
        #判断path文件夹是否存在，如果不存在则创建path文件夹
        if os.path.exists(path):
            continue
        else:
            os.mkdir(path)
        
        #生成波形图并保存
        plt.figure()
        plt.plot(sq)
        plt.savefig(path+'\\oscillogram.png')
        plt.close()
        
        #dim = Determine_dimension(sq)
        dim=10
        
        delay = Choose_delay(sq, dim)
        
        #做滑窗嵌入
        points = Delay_embedding(sq, delay, Dim = dim)
        
        if len(points) >= 3:
            #PCA后生成可视图像
            reduced_data = Use_PCA(points)
            plt.savefig(path+'\\3D.png')
            plt.close()
        
        diag = Persistence(points, Max_edge_length = 10.0, Max_dimension = 3)
        gudhi.plot_persistence_barcode(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True)
        plt.savefig(path+'\\barcode.png')
        plt.close()
        gudhi.plot_persistence_diagram(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True, greyblock=True)
        plt.savefig(path+'\\diagram.png')
        plt.close()
        
        plt.close('all')