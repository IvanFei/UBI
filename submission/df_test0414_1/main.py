# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
import sklearn
import scipy as sp
import time
import random
import xgboost
import warnings
warnings.filterwarnings("ignore")

# model 
from sklearn import linear_model, svm, ensemble, neighbors
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from xgboost import XGBRegressor

# 导入模型
from model import *

## 文件地址
path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

columns = ['TERMINALNO', 'TIME', 'TRIP_ID', 'LONGITUDE', 'LATITUDE', 'DIRECTION', 'HEIGHT', 'SPEED', 'CALLSTATE']
target = ['Y']

def read_csv(path, n_rows=None):
    """
    文件读取模块，头文件见columns.
    :return: 
    """
    # for filename in os.listdir(path_train):
    tempdata = pd.read_csv(path, nrows=n_rows)
    #tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
    #                   "CALLSTATE", "Y"]
    return tempdata


def process(pred):
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    
    count = 0
    with open(path_test) as lines:
        with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
            writer = csv.writer(outer)
            i = 0
            ret_set = set([])
            for line in lines:
                if i == 0:
                    i += 1
                    writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
                    continue
                item = line.split(",")
                if item[0] in ret_set:
                    continue
                # 此处使用随机值模拟程序预测结果
                #writer.writerow([item[0], np.random.rand()]) # 随机值
                writer.writerow([item[0], pred[count]]) # 随机值
                count += 1
                ret_set.add(item[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重
                


# range feature// feature extract
def rangeFeat(arr):
    
    return arr.max() - arr.min()

def calcStd(arr):
    
    if (arr.shape[0] == 1):
        return 0
    else:
        return arr.std()
    
def mode(arr):
    
    return sp.stats.mode(arr, axis=None)[0]

# preprocess
def dataXPreprocess(df):
    """
    数据预处理。
    特征的提取，归一化，数据的导入
    """
     ## number of sample
    terminalno = set(df[columns[0]])
    numSample = len(terminalno)
    print('Number of train/test data: {}'.format(numSample))
    numFeat = 10
    
    #print(df.isnull().sum())
    df[df == -1] = np.NAN
    print('Number of -1 value:')
    print(df.isnull().sum())
    df = df.fillna(method='bfill')
    
    # data and target
    dataX = df[columns[:]]
    
    #提取特征包括 mean/max/std/mode
    dataFeatExtract = dataX.groupby([columns[0]]).agg(
            {'TIME': {'sum': 'mean'}, 'TRIP_ID': {'max': 'max'}, 
             'LONGITUDE': {'mean': 'mean'}, 'LATITUDE': {'mean': 'mean'},
             'DIRECTION': {'std': calcStd}, 'HEIGHT': {'mean': 'mean', 'std': calcStd}, 
             'SPEED': {'mean': 'mean', 'std': calcStd}, 'CALLSTATE': {'mode':  mode}})
    
    #提取每个行程的时间跨度后加和为驾驶的时间
    Time = dataX.groupby([columns[0], columns[2]]).agg({'TIME': {'rangeFeat': rangeFeat}})
    T = np.zeros([numSample])
    for i, ind in enumerate(Time.index.levels[0]):
        T[i] = np.array(Time.xs(ind)).sum()
        
    dataFeatExtract['TIME'] = T
    
    #转换成ndarray 
    dataCollect = np.array(dataFeatExtract)
    
    # Normalization
    X = np.zeros([numSample, numFeat])
    X[:, :-1] = preprocessing.StandardScaler().fit_transform(dataCollect[:, :-1])
    X[:, -1] = dataCollect[:, -1]
    
    return X

def dataYPreprocess(df):
    
    #产生Y
    Y = np.array(df.groupby(columns[0], as_index=False).mean()['Y'])
    return Y

def modelFit(modelSet, X, Y):
    
    #训练模型集
    for rlf in modelSet:
        rlf[1].fit(X, Y)
    
    print("Model fit finished!!!")
    return modelSet

def modelPredict(modelSet, X):
    
    #预测
    numMode = len(modelSet)
    numSample = X.shape[0]
    pred = np.zeros([numSample, numMode])
    for i, rlf in enumerate(modelSet):
        pred[:, i] = rlf[1].predict(X)
    
    print("Model predict finished!!!") 
    return np.mean(pred, axis=1)
    

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    data = read_csv(path_train, n_rows=10000000) #6000000
    print(data.info())
    X = dataXPreprocess(data)
    Y = dataYPreprocess(data)
    #参数选择
    rlfSet = model(X, Y)
    # 训练模型
    rlfSet = modelFit(rlfSet, X, Y)
    testData = read_csv(path_test)
    testX = dataXPreprocess(testData)
    #预测
    pred = modelPredict(rlfSet, testX)
    
    process(pred)
