import math
import numpy as np
import pandas as pd
def Firefly_Clustering(data,cluster_label_list,beta,gamma):
    min_cost=9999999
    point_num=data.shape[0]
    # distance_matrix=Get_Distance_Matrix(data)
    original_cluster_center=Init_Cluster_Center(data,cluster_label_list) # 获得初始的簇中心
    light_list=Update_Light(data, cluster_label_list, original_cluster_center) # 获得每个点对应的亮度
    attractiveness_frame=np.zeros([point_num,point_num],"float") # 获得两个点之间的吸引力，得到吸引力列表
    for i in range(0,point_num):
        for j in range(0,point_num):
            attractiveness_frame[i][j]=Get_Attractiveness(data,i,j,beta,gamma)
    round = 0
    final_cluster_center=[]
    while True:
        for i in range(0,point_num):
            for j in range(0,point_num):
                if light_list[j]<light_list[i]:# 如果点j亮度比点i低
                    Move(data,i,j,attractiveness_frame)#点i向点j移动
                    attractiveness_frame[i][j]=Get_Attractiveness(data,i,j,beta,gamma)# 更新吸引力表格
                    attractiveness_frame[j][i]=Get_Attractiveness(data,j,i,beta,gamma)
                    cluster_center = Init_Cluster_Center(data, cluster_label_list)# 由于点进行了移动，因此簇中心也会发生改变，需要重新计算
                    light_list=Update_Light(data, cluster_label_list, cluster_center)# 更新亮度
        jk=Evaluating(data,cluster_label_list,cluster_center)# 评估是否找到最佳的簇中心
        round += 1
        # 终止条件
        if min_cost>jk:
            min_cost=jk
            np.savetxt("cluster_center.txt", cluster_center)
            final_cluster_center=cluster_center
        else:
            print("Firefly Clustering algorithm has performed about ",round," rounds")
            break
    return final_cluster_center



def Evaluating(data,cluster_label_list,cluster_center):
    """
    簇中心评估函数，计算每个点与对应簇中心之间的差距，这个值越小说明聚类效果越好
    :param data:
    :param cluster_label_list:
    :param cluster_center:
    :return:
    """
    jk=0
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            corresponding_cluster_center_label=cluster_label_list[i]# 找到对应簇中心序号
            jk+=abs(data[i][j]-cluster_center[corresponding_cluster_center_label][j]) # 计算该点与簇中心的欧氏距离

    return jk

def Get_Distance_Matrix(data):
    distance_matrix=np.zeros([data.shape[0],data.shape[0]])
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[0]):
            dis=0
            for k in range(0,data.shape[1]):
                dis+=(data[i][k]-data[j][k])**2
            dis=math.sqrt(dis)
            distance_matrix[i][j]=dis
    return distance_matrix

def Get_Attractiveness(data,i,j,beta,gamma):
    """
    计算吸引力
    :param data: 训练集
    :param i: 点i
    :param j: 点j
    :param beta: 输入参数
    :param gamma: 输入参数
    :return: 点i与点j之间的吸引力
    """


    r = 0
    for n in range(0, data.shape[1]):
        r += math.pow((data[i][n] - data[j][n]), 2)
    r=math.sqrt(r)
    for n in range(0, data.shape[1]):
        attractiveness=beta*math.e**(-gamma*math.pow(r,2))
    # print("点",i,"和点",j,"之间的吸引力是",attractiveness)
    return attractiveness

def Move(data,i,j,attractiveness_frame):
    """
    让点i向点j移动
    :param data: 训练集
    :param i: 点i
    :param j: 点j
    :param attractiveness_frame: 吸引力表格
    :return: 移动后的训练集
    """

    for n in range(0,data.shape[1]):
        data[i][n]=data[i][n]+attractiveness_frame[i][j]*(data[j][n]-data[i][n])
    return data

def Init_Cluster_Center(data,cluster_label_list):
    """
    初始化簇中心
    :param data:所有数据
    :param cluster_label_list: 每个点对应的label
    :return:
    """
    cluster_set=set(cluster_label_list)
    label_point_list=[]
    ck_list=[]# 簇中心的list ck_list[0]表示标签都是0的点的簇中心
    # 根据簇数目建立对应数量的list eg: 3个簇，则建立3个list，并且把这3个list放到一个大的list里面
    for i in range(0,len(cluster_set)):
        label_point_list.append([])
    # 按照每个点的label把该点放到对应的list里面 eg: label_point_list[0]表示标签都是0的点
    for i in range(0,data.shape[0]):
        now_label=cluster_label_list[i]
        label_point_list[now_label].append(data[i])
    # 获取对应list的簇中心
    for i in range(0, len(cluster_set)):
        ck=Get_Cluster_Center(np.array(label_point_list[i]))
        ck_list.append(ck)
    return  ck_list




def Get_Cluster_Center(data):
    """
    用于获取聚类的簇中心
    :param data:属于这个簇的点
    :return: 返回簇中心坐标
    """
    ck=[]
    for j in range(0, data.shape[1]):
        position=0
        for i in range(0,data.shape[0]):
            position+=data[i][j]
        position=position/data.shape[0]
        ck.append(position)
    return ck

def Update_Light(data,cluster_label_list,cluster_center):
    """
    代价函数(亮度函数)，获得每个点对应的亮度并且将其存储在light_list里面.亮度（代价）越低越好
    :param data:输入数据
    :param cluster_label_list:每个点对应的簇中心序号
    :param cluster_center: 每个簇中心的位置
    :return: 代价fi
    """
    light_list=[]
    for i in range(0,data.shape[0]):# 遍历每一个点
        sum = 0
        corresponding_cluster_center_label = cluster_label_list[i]  # 找到对应簇中心序号

        for j in range(0,data.shape[1]): # 计算该点与簇中心的欧式距离
            sum+=math.pow((data[i][j]-cluster_center[corresponding_cluster_center_label][j]),2) # 计算该点与簇中心的欧氏距离
        sum=math.sqrt(sum)
        fi=sum/data.shape[0]# 距离归一化
        light_list.append(fi)
    return light_list