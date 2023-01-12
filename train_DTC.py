# 模型搭建
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
import gensim
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os
import time
from sklearn import tree
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import joblib

from dataloader import *
from config import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 训练
def train():
    # 词向量初始化
    df = pd.read_csv(waimai_cut_path, header=None)
    print(df)
    
    '''
    sentences = df.iloc[:, 1].astype("str").map(lambda x: x.split(" "))
    #word2vec
    word_vec_model = gensim.models.Word2Vec(sentences, vector_size=128, workers=4, min_count=0)
    # 构建词向量
    word_index, embeddings_matrix = build_embeddings_matrix(word_vec_model)
    # 拆分数据集
    x_train, x_test, y_train,y_test = get_data_DTC(word_index)
    print(x_train)
    
    # 从训练集上分出测试集
    sentences = df.iloc[:, 1].astype("str")
    x_train, x_test, y_train, y_test = train_test_split(sentences, df.iloc[:, 0].values, test_size=0.2, random_state=1)

    
    #利用词频进行特征提取
    count_vec = CountVectorizer(binary=True)
    x_train = count_vec.fit_transform(x_train)
    print(x_train)
    x_test = count_vec.transform(x_test)#为啥不是fit_transform
    print(x_train)
    
    '''
    # 从训练集上分出测试集
    sentences = df.iloc[:, 1].astype("str")
    x_train, x_test, y_train, y_test = train_test_split(sentences, df.iloc[:, 0].values, test_size=0.2)#, random_state=1
    #TF-IDF
    vectorizer = TfidfVectorizer()
    vectorizer.fit(x_train)
    x_train=vectorizer.transform(x_train)
    x_test=vectorizer.transform(x_test)
    print(x_train)
    start_time = time.time()
    
    # 构建决策树模型
    model_dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=10)#criterion为分裂节点所用的标准
    
    # 进行训练
    model_dtc.fit(x_train, y_train)
    end_time = time.time()
    print('Sum time is %f' % (end_time - start_time))
    print('在训练集上的准确率：%.5f' % accuracy_score(y_train, model_dtc.predict(x_train)))
    y_true = y_test
    y_pred = model_dtc.predict(x_test)
    print('在测试集上的准确率：%.5f' % accuracy_score(y_true, y_pred))
    # 保存模型
    joblib.dump(model_dtc, model_path_DTC)
    compute_indexes(y_test,y_pred)
    
# 计算常用指标
def compute_indexes(y_test, y_test_pred):
    print(classification_report(y_test, y_test_pred))

    #混淆矩阵 
    cm = confusion_matrix(y_test, y_test_pred)
    df=pd.DataFrame(cm,index=["0", "1"],columns=["0", "1"])
    sns.heatmap(df,annot=True,fmt="d")
    plt.savefig('./img/DTC_confusion_matrix.png')
    print(cm)
    tp,fp,fn,tn=cm[0][0],cm[0][1],cm[1][0],cm[1][1]
    
    #常用指标
    accuracy = (tp+tn) / (tp+tn+fp+fn)     # 准确率
    precision = tp / (tp+fp)               # 精确率
    recall = tp / (tp+fn)                  # 召回率
    F1 = (2*precision*recall) / (precision+recall)    # F1
    print("Accuracy:  {:.4f}%".format(accuracy*100))
    print("Precision: {:.4f}%".format(precision*100))
    print("Recall:    {:.4f}%".format(recall*100))
    print("F1:        {:.4f}%".format(F1*100))


if __name__ == '__main__':
    train()