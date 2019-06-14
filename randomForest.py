#coding:utf-8

from math import log
import math
import numpy as np
import pandas as pd
import operator
from sklearn.model_selection import train_test_split

np.set_printoptions(precision = 3)      # 设置numpy输出的小数为3位, 否则精度会非常大, 导致溢出


# 通过排序返回出现次数最多的类别
def majorityCnt(trainLabel):
    labelCount = {}             # 统计每个类别的个数
    for i in trainLabel:
        if i not in labelCount.keys(): 
            labelCount[i] = 0
        labelCount[i] += 1
    sortedlabelCount = sorted(labelCount.items(), key=operator.itemgetter(1), reverse=True) # 将类别数降序排列
    return sortedlabelCount[0][0]    


# 计算信息熵
def calEnt(trainLabel):
    numEntries = len(trainLabel)            # 样本数D
    labelCount = {}
    for i in trainLabel:  
        if i not in labelCount.keys():      # 统计每个类别的个数
            labelCount[i] = 0
        labelCount[i] += 1
    Ent = 0.0
    for key in labelCount:  
        p = float(labelCount[key]) / numEntries
        Ent = Ent - p * log(p, 2)
    return Ent


# 划分数据集, 参数为数据集、划分特征、划分值 
def splitDataSet(trainData, trainLabel, feature, value):
    trainDataLeft = []
    trainDataRight= []
    trainLabelLeft = []
    trainLabelRight = []

    for i in range(len(trainData)):
        if float(trainData[i][feature]) <= value:
            trainDataLeft.append(trainData[i])
            trainLabelLeft.append(trainLabel[i])
        else:
            trainDataRight.append(trainData[i])
            trainLabelRight.append(trainLabel[i])

    return trainDataLeft, trainDataRight, trainLabelLeft, trainLabelRight 


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(trainData, trainLabel):

    featureNum = len(trainData[0])          # 特征数
    baseEnt = calEnt(trainLabel)            # 计算信息熵
    bestGainRatio = 0.0
    bestFeature = -1
    bestPartValue = None 
    IV = 0.0

    for feature in range(featureNum):           # 对每个特征, 计算信息增益率
        featList = [i[feature] for i in trainData]
        uniqueVals = set(featList)              # 去重
        bestPartValuei = None

        sortedUniqueVals = list(uniqueVals)     # 升序排列
        sortedUniqueVals.sort()

        minEnt = float("inf")
        for i in range(len(sortedUniqueVals) - 1):      
            partValue = (sortedUniqueVals[i] + sortedUniqueVals[i + 1]) / 2             # 计算划分点

            (trainDataLeft, trainDataRight, trainLabelLeft, trainLabelRight) = splitDataSet(trainData, trainLabel, feature, partValue)     # 对每个划分点, 计算ΣEnt(D^v)
            pLeft = len(trainDataLeft) / float(len(trainData))
            pRight = len(trainDataRight) / float(len(trainData))
            Ent = pLeft * calEnt(trainLabelLeft) + pRight * calEnt(trainLabelRight)     # 计算ΣEnt(D^v)
   
            
            if Ent < minEnt:        # ΣEnt(D^v)越小, 则信息增益Gain = Ent(D) - ΣEnt(D^v)越大
                minEnt = Ent
                IV = -(pLeft * log(pLeft, 2) + pRight * log(pRight, 2))                 # 计算IV
                bestPartValuei = partValue

        Gain = baseEnt - minEnt     # 计算信息增益Gain
        GainRatio = Gain / IV       # 计算信息增益率GainRatio

        if GainRatio > bestGainRatio:       # 取最大的信息增益率对应的特征
            bestGainRatio = GainRatio
            bestFeature = feature
            bestPartValue = bestPartValuei

    return bestFeature, bestPartValue


# 创建树
# @params list类型的m*n样本, list类型的1*m分类标签, 决策树最大深度, 决策树节点所含最少样本数, 当前决策树深度
def createTree(trainData, trainLabel, max_depth, min_size, depth):
    if trainLabel.count(trainLabel[0]) == len(trainLabel):                          # 如果只有一个类别，返回该类别
        return {'label': trainLabel[0]}

    if len(trainLabel) <= min_size:                                                 # 如果样本数 <= 节点所含最少样本数, 返回出现次数最多的样本
        return {'label': majorityCnt(trainLabel)}

    if depth >= max_depth:                                                          # 如果决策树深度 >= 决策树最大深度, 返回出现次数最多的样本
        return {'label': majorityCnt(trainLabel)}

    bestFeat, bestPartValue = chooseBestFeatureToSplit(trainData, trainLabel)     # 获取最优划分特征的索引, 以及该特征的划分值
    myTree = {'feature': bestFeat, 'value': bestPartValue}

    (trainDataLeft, trainDataRight, trainLabelLeft, trainLabelRight) = splitDataSet(trainData, trainLabel, bestFeat, bestPartValue)


    # 构建左子树
    myTree['leftTree'] = createTree(trainDataLeft, trainLabelLeft, max_depth, min_size, depth + 1)
    # 构建右子树
    myTree['rightTree'] = createTree(trainDataRight, trainLabelRight, max_depth, min_size, depth + 1)
    return myTree


# 测试算法
def classify(myTree, testData):
    if 'label' in myTree.keys():        # 叶节点
        return myTree['label']
    feature = myTree['feature']
    partValue = myTree['value']
    if testData[feature] <= partValue:
        return classify(myTree['leftTree'], testData)
    else: 
        return classify(myTree['rightTree'], testData)



# 后剪枝
def postPruningTree(myTree, trainData, trainLabel):

    if 'label' in myTree.keys():    # 叶节点
        return myTree

    (trainDataLeft, trainDataRight, trainLabelLeft, trainLabelRight) = splitDataSet(trainData, trainLabel, myTree['feature'], myTree['value'])
  
    myTree['leftTree'] = postPruningTree(myTree['leftTree'], trainDataLeft, trainLabelLeft)             # 对左子树剪枝
    myTree['rightTree'] = postPruningTree(myTree['rightTree'], trainDataRight, trainLabelRight)         # 对右子树剪枝

    predict = []                            # 预测结果
    for i in trainData:
        predict.append(classify(myTree, i))

    majorLabel = majorityCnt(trainLabel)    # 选取最多的类别来进行剪枝判断

    error1 = 0.0
    error2 = 0.0

    for i in range(len(trainLabel)):        # 计算剪枝与不剪枝的误差
        error1 = error1 + (predict[i] - trainLabel[i])**2
        error2 = error2 + (majorLabel - trainLabel[i])**2

    if error1 <= error2:                    # 若不剪枝误差小
        return myTree

    return {'label': majorityCnt(trainLabel)}   
    


if __name__ == '__main__':
 
    last_trainData = pd.DataFrame()
    last_trainLabel = pd.DataFrame()

    for n in range(1, 6):               # 读取训练数据
        
        trainData_reader = pd.read_csv('train' + str(n) + '.csv', encoding = 'utf-8', header = None, iterator = True)
        trainLabel_reader = pd.read_csv('label' + str(n) + '.csv', encoding = 'utf-8', header = None, iterator = True)

        i = 0

        while True:
            try:
                print("{} {}".format(n, i))
                i+=1
                trainData = trainData_reader.get_chunk(100000)
                last_trainData = pd.concat([last_trainData, trainData])

                trainLabel = trainLabel_reader.get_chunk(100000)
                last_trainLabel = pd.concat([last_trainLabel, trainLabel])

            except StopIteration:
                break 
                
            del trainData      # 释放内存
            del trainLabel
 
    last_trainData = last_trainData.values                      # dataFrame转np.array
    last_trainLabel = last_trainLabel.values.ravel()
    
    max_depth = 8                                               # 决策树最大深度
    min_size = 1                                                # 决策树节点所含最少样本数
    tree_number = 1000                                          # 随机森林中决策树的个数

    randomForest = []

    for i in range(tree_number):
        X, trainData, y, trainLabel = train_test_split(last_trainData, last_trainLabel, test_size = 0.001)          # 获取1/1000的训练数据  
        X, verifyData, y, verifyLabel = train_test_split(last_trainData, last_trainLabel, test_size = 0.001)        # 获取1/1000的验证数据  
        myTree = createTree(trainData, trainLabel, max_depth, min_size, 1)          # 生成决策树
        myTree = postPruningTree(myTree, verifyData, verifyLabel)                   # 后剪枝
        randomForest.append(myTree)                                                 # 放入森林

    print('-----------------------------------------------------------')
    
    
    last_testData = pd.DataFrame()
    for n in range(1, 7):

        testData_reader = pd.read_csv('test' + str(n) + '.csv', encoding = 'utf-8', header = None, iterator = True)  # 读取testData, 并返回DataFrame

        while True:
            try:
                testData = testData_reader.get_chunk(100000)
                last_testData = pd.concat([last_testData, testData])

            except StopIteration:
                break

            del testData
        del testData_reader

    testData = last_testData.values

    predict = []                                                                # 预测结果
    for i in testData:
        sum = 0
        for tree in randomForest:                                               # 对随机森林中每棵树进行预测, 然后取平均值
            sum += classify(tree, i)
        
        mean = sum / tree_number                                                
        predict.append(mean)

    
    sub = pd.DataFrame(predict)    
    sub.columns = ['Predicted']
    sub.insert(0, 'id', [i for i in range(1, len(predict) + 1)])                # 插入id列
    sub.to_csv('./submit.csv', index = 0, encoding = "utf-8", mode='a')         # index=0表示不保留行索引, mode='a'表示追加写入, header = 0表示不写入列索引

    
    del last_testData
    del predict
    
    print("ok")