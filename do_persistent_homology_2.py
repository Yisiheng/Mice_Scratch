# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 21:29:41 2022

@author: h'h
"""

import do_video
import do_embedding
import numpy as np
import matplotlib.pyplot as plt
import gudhi
from gudhi.point_cloud import timedelay
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

#Str='E:\\小鼠\\negative\\negative\\202206021135-1_02-167-51550-51685.mkv'
#Str='E:\\小鼠\\positive\\positive\\202206020911-0_04-30-10002-10226.mkv'


#g_i 只考虑对整个视频进行滑窗嵌入，某个函数会利用PCA_image做简单的降维，某个函数会考虑马赛克操作降维，某个函数会考虑直接降噪且去除背景

def g_1(Path):
    #对单个视频的完整数据做滑窗嵌入

    XOrig,shape=do_video.Read_video(Path, Key = 1)
    dim=7
    X = do_embedding.getPCAVideo(XOrig)
    (x, maxT, corr) = do_embedding.fundamentalFreqEstimation(X)
    Tau = maxT/float(dim)
    
    #M = X.shape[0] - maxT + 1
    #dT = M/float(desiredSamples)
    
    dT=1
    XS = do_embedding.getSlidingWindowVideo(X, dim, Tau, dT)
    XS = XS - np.mean(XS, 1)[:, None]
    XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
    D = do_embedding.getSSM(XS)
    
    reduced_data=do_embedding.Use_PCA(XS)
    
    rips_complex=gudhi.RipsComplex(distance_matrix = D)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension = 3)

    diag=simplex_tree.persistence(homology_coeff_field = 11, min_persistence = 0.001, persistence_dim_max = False)
    gudhi.plot_persistence_barcode(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True)
    
    return diag




def g_2(Path):
    #没有中心化，没有正规化，没有做PCA
    #考虑了多个Tau的取值，但因为数据量小，所以Tau值不能取太大
    
    XOrig,shape=do_video.Read_video(Path, Key = 1)
    
    dim=5
    dT=1
    for i in range(10):
        Tau = i + 1
        
        XS = do_embedding.getSlidingWindowVideo(XOrig, dim, Tau, dT)
        
        '''
        XS = XS - np.mean(XS, 1)[:, None]
        XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
        '''
        
        reduced_data=do_embedding.Use_PCA(XS)
        fig = plt.figure('Tau:'+str(Tau))
        ax = Axes3D(fig)
        ax.scatter(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2])
        ax.plot3D(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2], 'gray')
        
        '''
        rips_complex=gudhi.RipsComplex(XS)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension = 3)
        
        diag=simplex_tree.persistence(homology_coeff_field = 3, min_persistence = 0.001, persistence_dim_max = False)
        gudhi.plot_persistence_barcode(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True)
        '''

def g_3(X):
    #
    
    dim=7
    X = do_embedding.getPCAVideo(X)
    (x, maxT, corr) = do_embedding.fundamentalFreqEstimation(X)
    Tau = maxT/float(dim)
    M = X.shape[0] - maxT + 1
    dT=1
    XS = do_embedding.getSlidingWindowVideo(X, dim, Tau, dT)
    XS = XS - np.mean(XS, 1)[:, None]
    XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
    D = do_embedding.getSSM(XS)

    reduced_data=do_embedding.Use_PCA(XS)

    rips_complex=gudhi.RipsComplex(distance_matrix = D)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension = 3)

    diag=simplex_tree.persistence(homology_coeff_field = 3, min_persistence = 0.001, persistence_dim_max = False)
    gudhi.plot_persistence_barcode(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True)
    
    return diag

def g_4(Path, Score = 0.95):
    #做PCA降维后再考虑滑窗嵌入与持续同调
    #Str='E:\\小鼠\\positive\\positive\\202206020911-1_02-29-5381-5555.mkv'
    
    XOrig,shape=do_video.Read_video(Path, Key = 1)
    
    #保留Score值的贡献度
    pca = PCA(Score)
    reduced_data = pca.fit_transform(XOrig)
    #data_pca=pca.inverse_transform(reduced_data)
    #plt.figure("PCA_image")
    #plt.imshow(data_pca[1])
    
    dim=5
    Tau=10
    dT=1
    Points=do_embedding.getSlidingWindowVideo(reduced_data, dim, Tau, dT)
    
    #做可视化操作，通过PCA降维至3维空间显示
    pca = PCA(n_components = 3)
    reduced_points = pca.fit_transform(Points)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(reduced_points[:,0],reduced_points[:,1],reduced_points[:,2])
    ax.plot3D(reduced_points[:,0],reduced_points[:,1],reduced_points[:,2], 'gray')
    
    
    rips_complex=gudhi.RipsComplex(Points)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension = 3)
    
    diag=simplex_tree.persistence(homology_coeff_field = 3, min_persistence = 0.001, persistence_dim_max = False)
    gudhi.plot_persistence_barcode(persistence=diag, max_intervals=100, inf_delta=0.1, legend=True)











def g_5(Path):
    #先对原始视频执行打马赛克类似的操作，一为降维，二为去噪
    
    XOrig,shape=do_video.Read_video(Path, Key = 0)
    
    msize=(20, 20)
    
    Nframe, length, width= XOrig.shape
    
    X=np.zero((Nframe, length*width))
    
    for i in range(Nframe):
        
        X[i,:]=do_video.Mosaic(XOrig[i], msize)










