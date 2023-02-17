# -*- coding: utf-8 -*-
"""
@Time ： 2023/2/17 22:02
@Auth ： Shijie Zeng
@GitHub：https://github.com/firesaku/
"""
import Train
import numpy as np
import Test
from sklearn.model_selection import train_test_split


# 代码复刻了文献：Senthilnath J, Omkar S N, Mani V. Clustering using firefly algorithm: performance study[J]. Swarm and Evolutionary Computation, 2011, 1(3): 164-171.
# doi: 10.1016/j.swevo.2011.06.003
# 这篇论文的萤火虫聚类（FC）算法属于有监督聚类。 需要一定数据进行预训练得到簇中心，然后将测试集数据分配到指定的簇上。


# Train
datafile = np.loadtxt("datasets/iris.csv", delimiter=",")# 读取文件
train_set, test_set = train_test_split(datafile, test_size=0.2, random_state=123) # 划分测试集和训练集。 test_size为测试集大小， random_state为随机种子
data=train_set[:,0:-1]# 获取训练集数据
label=train_set[:,-1].astype(int).tolist() # 获取训练集标签

beta=1 # FC模型参数
gamma=0.5 # FC模型参数
final_cluster_center=Train.Firefly_Clustering(data,label,beta,gamma)# 返回训练好的聚类中心，簇中心数据也会保存到 ' cluster_center.txt' 中

# Test
data=test_set[:,0:-1] # 获取测试集数据
label=test_set[:,-1].astype(int).tolist() # 获取测试集标签
# cluster_center = np.loadtxt("cluster_center.txt", delimiter=" ") # 通过文件读取测试集数据
cluster_result=Test.FC_Test(data,np.array(final_cluster_center)) #得到测试集的预测结果

# 评估模型
ac_rate=0
cluster_result=cluster_result.tolist()
for i in range(0,len(cluster_result)):
    if cluster_result[i]==label[i]:
        ac_rate+=1
ac_rate=ac_rate/data.shape[0]*100
print(label) # 测试集标签
print(cluster_result)# 测试集的预测结果
print(ac_rate) # 测试集正确率