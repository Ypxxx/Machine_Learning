import numpy as np
import matplotlib.pyplot as plt
import operator
from math import log
from math import exp


############################################################ Normalization ###############################################################################
def normalization(dataset):
    minvalue = dataset.min(0)
    maxvalue = dataset.max(0)
    range = maxvalue - minvalue
    normaldata = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normaldata = dataset-np.tile(minvalue,(m,1))
    normaldata = normaldata/(np.tile(range,(m,1)))
    return normaldata


############################################################ KNNClassfier #################################################################################
def classifydata():
    group = np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels = ["A","A","B","B"]
    return group,labels
def KNNclassify(inX,x,y,k):
    dataSetSize = x.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - x #tiel 复制列表
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    sqDistances = sqDistances**0.5
    sortedDistIndicies = sqDistances.argsort() #返回从小到大的索引
    classCount={}
    for i in range(k):
        voteIlabel = y[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #operator.itemgetter(1)每次获取第二个元素，按第二个元素进行排序
    return sortedClassCount[0][0]

######################################################################################### KNNregression ###############################

def regressdata():
    group = np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels = [1,1.5,2,2.5]
    return group,labels

def KNNregress0(inX,x,y,k):
    dataSetSize = x.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - x
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    sqDistances = sqDistances**0.5
    sortedDistIndicies = sqDistances.argsort()
    classCount={}
    d=[]
    for i in range(k):
        d.append(y[sortedDistIndicies[i]])
    #     voteIlabel = y[sortedDistIndicies[i]]
    #     classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # return sortedClassCount[0][0]
    return sum(d)/k
################################################################################################################################

############################################################### Decision Tree #################################################


dataset = [[1,1],[1,1],[1,0],[0,1],[0,1]]
y = ["y","y","n","n","n"]
labels = ["no surfacing","flippers"]

for i in range(len(y)):
    dataset[i].extend(y[i])
def calcShannonEnt(dataset):

    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def splitDataSet(dataset,axis,value):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #保留之前的数据
            reducedFeatVec.extend(featVec[axis+1:]) #保留之后的数据
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseVsetFeatureToSplit(dataset):
    numFeaures = len(dataset[0])-1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain=0.0
    bestFeature = -1
    for i in range(numFeaures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList) ## 消除重复元素
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset,i,value)
            prob = len(subDataSet)/float(len(dataset))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majoritycnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(dataset,labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majoritycnt(classList)
    bestFeat = chooseVsetFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset,bestFeat,value),subLabels)
    return myTree

decisionNode = dict(boxstyle="sawtooth",fc="0.8")
leafNode = dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords="axes fraction",xytext=centerPt,textcoords="axes fraction"
                            ,va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1,facecolor="w")
    fig.clf()
    createPlot().ax1 = plt.subplot(111,frameon=False)
    plotNode("决策节点",(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode("叶节点",(0.8,0.1),(0.3,0.8),leafNode)
    plt.show

######################################################################################### LogisticeRegression ############################################
def loadDataSet():
    dataMat =[];
    labelMat =[]
    fr = open("testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inx):
    return 1.0/(1+exp(-inx))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n =np.shape(dataMatrix)
    alpha = 0.001
    maxcycles = 500
    weights = np.ones((n,1))
    for k in range(maxcycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights

def stocGradAscent0(dataMatrix,classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights+alpha*error*dataMatrix[i]
    return weights

##################################################################################
