#-*- coding:utf-8 -*-
import re
import xlrd
import xdrlib,sys
import xlwt
import datetime
import  time
import numpy as np
def rand(a, b): #返回a 行 n 列 （0,1）的随机数矩阵

    A=np.zeros((a,b))
    from random import random
    for h in range(a):
        for l in range(b):
            A[h][l]=random()
    return A
if __name__=="__main__":

    #异或的数据
    data=np.array([[0,0],[0,1],[1,0],[1,1]])
    y=np.array([0,1,1,0])

    t=10 #隐藏神经元的数目，大于输入层数目
    p=rand(4,t)       #径向基函数的值

    ty=rand(4,1)      #输出值

    w=rand(1,t)  #隐藏层第i个神经元与输出神经元的权重

    b=rand(1,t)  #样本与第i个神经元中心的距离的缩放系数

    tk=0.5

    # [id, c] = kmeans(x, 2);
    c = rand(t, 2) # 隐层第i个神经元的中心

    kn=0  #迭代次数
    sn=0  #累积误差值累积次数
    old_ey=0  #前一次迭代的累积误差
    print((np.array(data[0]) - np.array(c[0])).T)
    while 1:
        kn=kn+1

        #计算每个样本的径向基函数的值
        for i in range(4):
            for j in range(t):
                ij=(data[i]-c[j])
                ij=ij.dot(ij.T) #矩阵相乘
                p[i][j]=np.exp(-1*b[0][j]*ij)
            ty[i]=w.dot(p[i].T)
        print(p)

        #计算累积误差
        ey=(ty.T-y).dot((ty.T-y).T)
        ey=ey[0][0]
        print('ey:',ey)
        #g更新w,b
        dw=np.zeros((1,t))
        db=np.zeros((1,t))

        for i in range(4):
            for j in range(t):
                dw[0][j]=dw[0][j]+(ty[i]-y[i])*p[i][j]
                ij = (data[i] - c[j])
                ij = ij.dot(ij.T)  # 矩阵相乘
                db[0][j]=db[0][j]-(ty[i]-y[i])*w[0][j]*ij*p[i][j]
        print("dw:", dw)
        w=w-tk*dw/4
        b=b-tk*db/4

        #迭代终止条件
        if(abs(old_ey-ey)<0.0001):
            sn=sn+1
            if(sn==10):
                break
        else:
            old_ey=ey
            sn=0
