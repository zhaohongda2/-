# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import time
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier



#read data
print("reading data")
dataset = pd.read_csv(r'E:\kaggle\shuzi-data\train.csv')
X_train = dataset.values[0:, 1:]
y_train = dataset.values[0:, 0]


"""#for fast evaluation
X_train_small = X_train[:10000, :]
y_train_small = y_train[:10000]
#knn
#-----------------------用于小范围测试选择最佳参数-----------------------------#
#begin time
start = time.clock()

#progressing
print "selecting best paramater range"
knn_clf=KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', weights='distance', p=3)
score = cross_val_score(knn_clf, X_train_small, y_train_small, cv=3)

print( score.mean() )
#end time
elapsed = (time.clock() - start)
print("Time used:",int(elapsed), "s")

#k=3
#0.942300738697
#0.946100822903 weights='distance'
#0.950799888775 p=3
#k=5
#0.939899237556
#0.94259888029
#k=7
#0.935395994386
#0.938997377902
#k=9
#0.933897851978


X_test = pd.read_csv(r'E:\kaggle\shuzi-data\test.csv').values

#from KNN.KNN_choosePara import *

clf=KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', weights='distance', p=3)

start=time.clock()



#read data
print("reading data")
clf.fit(X_train,y_train) #针对整个训练集训练分类器
elapsed = (time.clock() - start)
print("Training Time used:",int(elapsed/60) , "min")



print("predicting")
result=clf.predict(X_test)
result=np.c_[range(1,len(result)+1),result.astype(int)] #转化为int格式生成一列
df_result=pd.DataFrame(result,columns=['ImageID','Label'])

df_result.to_csv(r'E:\kaggle\shuzi-data\results.knn.csv',index=False)

#end time
elapsed = (time.clock() - start)
print("Test Time used:",int(elapsed/60) , "min")
