#Ziman David zdim1981

import math
from collections import defaultdict
import operator
import numpy as np

k = 10
LearntNumbers = []
TestNumbers = []
TestNumbersCount = [178, 182, 177, 183, 181, 182, 181, 179, 174, 180]
LearntNumbersCount = [376, 389, 380, 389, 387, 376, 377, 387, 380, 382]

#EUCLID DISTANCE
def Euclid(x, y):
    res = 0.0
    for i in range(0, 64):
        res += (x[i] - y[i])**2
    return math.sqrt(res)

#K_NEAREST_NEIGHBOUR
def kNN(img, k, nums):
    distances = []
    stat = [0,0,0,0,0,0,0,0,0,0,0]
    for i in range( 0,len(nums) ):
        distances.append( (Euclid(img, nums[i]), nums[i][64]) )
    distances.sort(key = operator.itemgetter(0))
    N_Neighbour = distances[:k]
    for i in range(0,len(N_Neighbour)):
        stat[int(N_Neighbour[i][1])] += 1
    max = -math.inf
    for i in range(10):
        if stat[i] > max :
            max = stat[i]
            index = i
    return index

#CENTROID OF IMG
def Centroid(img, nums):
    aux = np.zeros(64)
    mids = [aux, aux, aux, aux, aux, aux, aux, aux, aux, aux]
    counter = np.zeros(10)
    for i in nums:
        index = int(i[64])
        mids[index] = i[:-1] + mids[index]
        counter[index] += 1
    centroids = np.zeros((10, 64))
    for i in range(10):
        centroids[i] = mids[i] / counter[i]
    min = math.inf
    index = -1
    for i in range(10):
        if min > Euclid(centroids[i], img):
            min = Euclid(centroids[i], img)
            index = i
    return index

def Gradient(label1, label2):
    LIMIT = 0.00001
    GAMMA = 0.0001
    x = np.empty((0,64))
    y = np.empty((0,1))
    for tomb in LearntNumbers:
        if tomb[64] == label1 or tomb[64] == label2:
            x = np.vstack((x,tomb[:-1]))
            if tomb[64] == label1:
                y = np.vstack((y, np.array([1])))
            if tomb[64] == label2:
                y = np.vstack((y, np.array([-1])))
    l = len(y)
    x = np.hstack((x,np.ones((l,1))))
    xt = x.transpose()
    w = np.array([np.mean(x, axis=0)]).T
    prev_w = w - 1
    while np.linalg.norm(prev_w - w) > LIMIT:
        prev_w = w
        w = w - ((2*GAMMA) / l) * np.matmul(xt,np.matmul(x, w)-y)
    return w

def LinRegression(label1, label2):
    Lambda = 0.5
    x = np.empty((0,64))
    y = np.empty((0,1))
    for tomb in LearntNumbers:
        if tomb[64] == label1 or tomb[64] == label2:
            x = np.vstack((x,tomb[:-1]))
            if tomb[64] == label1:
                y = np.vstack((y,np.array([1])))
            if tomb[64] == label2:
                y = np.vstack((y,np.array([-1])))
    l = len(y)
    x = np.hstack((x,np.ones((l,1))))
    xt = x.transpose()
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(xt,x) + Lambda*np.identity(65)),xt),y)

def GradTest(label1, label2):
    correct1 = 0
    count1 = 0
    correct2 = 0
    count2 = 0
    w = Gradient(label1, label2)

    for i in TestNumbers:
        if i[64] == label1 or i[64] == label2:
            x = i[:-1]
            x = np.append(x,[1])
            answer = np.matmul(x,w)
            if i[64] == label1:
                count1 += 1
                if answer > 0:
                    correct1 += 1
            if i[64] == label2:
                count2 += 1
                if answer < 0:
                    correct2 += 1
    print("Gradient: ")
    print(label1, ' count: ', correct1, ' / ', count1)
    print('Error:', (1 - correct1 / count1) * 100, '%')
    print(label2, ' count: ', correct2, ' / ', count2)
    print('Error:', (1 - correct2 / count2) * 100, '%')
    print('All results: ', count1 + count2)
    print('All Error:', (1 - (correct1 + correct2) / (count1 + count2)) * 100, '%\n')

def LinRegTest(label1, label2):
    correct1 = 0
    count1 = 0
    correct2 = 0
    count2 = 0
    w = LinRegression(label1, label2)

    for i in TestNumbers:
        if i[64] == label1 or i[64] == label2:
            x = i[:-1]
            x = np.append(x,[1])
            answer = np.matmul(x,w)
            if i[64] == label1:
                count1 += 1
                if answer > 0:
                    correct1 += 1
            if i[64] == label2:
                count2 += 1
                if answer < 0:
                    correct2 += 1
    print("Linear Regression: ")
    print(label1, ' count: ', correct1, ' / ', count1)
    print('Error:', (1 - correct1 / count1) * 100, '%')
    print(label2, ' count: ', correct2, ' / ', count2)
    print('Error:', (1 - correct2 / count2) * 100, '%')
    print('All results: ', count1 + count2)
    print('All Error:', (1 - (correct1 + correct2) / (count1 + count2)) * 100, '%\n')

with open("optdigits.tes", "r") as f:
    for line in f:
        v = []
        elements = line.split(",")
        for i in elements:
            v.append(float(i.split("\n")[0]))
        TestNumbers.append(v)

with open("optdigits.tra", "r") as f:
    for line in f:
        v = []
        elements = line.split(",")
        for i in elements:
            v.append(float(i.split("\n")[0]))
        LearntNumbers.append(v)

def kNNTest(k):
    print('Knn on Test numbers:')
    aux = np.zeros(10)
    for i in TestNumbers:
        val = kNN(i, k, LearntNumbers)
        if val == i[64]:
            aux[int(i[64])] += 1
    for i in range(10):
        print(i, ' :', 100 - aux[i]*100/TestNumbersCount[i], '%')

def kNNLearn(k):
    print('kNN on Learnt numbers:')
    aux = np.zeros(10)
    for i in LearntNumbers:
        val = kNN(i, k, LearntNumbers)
        if val == i[64]:
            aux[int(i[64])] += 1
    for i in range(10):
        print(i, ' :', 100 - aux[i]*100/LearntNumbersCount[i], '%')

def CentroidTest():
    print('Centroid on Test numbers:')
    aux = np.zeros(10)
    for i in TestNumbers:
        val = Centroid(i, LearntNumbers)
        if val == i[64]:
            aux[int(i[64])] += 1
    for i in range(10):
        print(i, ' :', 100 - aux[i]*100/TestNumbersCount[i], '%')

def CentroidLearn():
    print('Centroid on Learnt numbers:')
    aux = np.zeros(10)
    for i in LearntNumbers:
        val = Centroid(i, LearntNumbers)
        if val == i[64]:
            aux[int(i[64])] += 1
    for i in range(10):
        print(i, ' :', 100 - aux[i]*100/LearntNumbersCount[i], '%')

# kNNLearn(k)
# kNNTest(k)
# CentroidLearn()
# CentroidTest()
GradTest(4, 7)
LinRegTest(4, 7)
# GradTest(5, 9)
# LinRegTest(5, 9)

# GradTest(6, 3)
# LinRegTest(6, 3)

# GradTest(3, 8)
# LinRegTest(3, 8)