import math
import numpy as np

def FC_Test(data,cluster_center):
    cluster_number=cluster_center.shape[0]
    cluster_result=np.zeros([data.shape[0]]).astype(int)
    for i in range(0,data.shape[0]):
        dis=999999
        for j in range(0,cluster_number):
            new_dis=Get_Distance(data[i],cluster_center[j])
            if new_dis<dis:
                dis=new_dis
                cluster_result[i]=j
    return cluster_result



def Get_Distance(data1,data2):
    dis=0
    for n in range(0,data1.shape[0]):
        dis=(data1[n]-data2[n])**2
    dis=math.sqrt(dis)
    return dis