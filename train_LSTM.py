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
import os
import time
from model_LSTM import *
from dataloader import *
from config import *
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
from sklearn.metrics import classification_report

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 训练
def train():
    # 打开文件
    df = pd.read_csv(waimai_cut_path, header=None)
    sentences = df.iloc[:, 1].astype("str").map(lambda x: x.split(" "))
    # 初始化词向量
    '''
    vector_size (int, optional) – word向量的维度。
    min_count (int, optional) – 忽略词频小于此值的单词。
    workers (int, optional) – 训练模型时使用的线程数。
    '''
    word_vec_model = gensim.models.Word2Vec(sentences, vector_size=128, workers=4, min_count=0)
    # 构建词向量
    word_index, embeddings_matrix = build_embeddings_matrix(word_vec_model)
    # 拆分数据集
    x_train, x_val, x_test, y_train, y_val, y_test = get_data_LSTM(word_index)
    print(x_train)
    # 开始计时
    start_time = time.time()
    # 构建模型
    model = LSTM(word_index, embeddings_matrix)
    # 训练    #一个epoch表示： 所有的数据送入网络中，完成了一次前向计算 + 反向传播的过程。
    history=model.fit(x_train, y_train, epochs=TOTAL_EPOCH, validation_data=(x_val, y_val))
    # 结束计时
    end_time = time.time()
    # 统计计时
    print('Sum time is %f' % (end_time - start_time))
    # 评估,在测试集上进行效果评估
    results = model.evaluate(x_test, y_test)
    print(f"损失: {results[0]}, 准确率: {results[1]}")
    # 保存模型
    model.save(model_path_LSTM)
    y_test_pred=(model.predict(x_test) > 0.5).astype("int32")
    print_history(history)
    compute_indexes(y_test, y_test_pred)

def print_history(history):
    #  使用history将训练集和测试集的loss和acc调出来
    acc = history.history['accuracy']  # 训练集准确率
    val_acc = history.history['val_accuracy']  # 测试集准确率
    loss = history.history['loss']  # 训练集损失
    val_loss = history.history['val_loss']  # 测试集损失
    #  打印acc和loss，采用一个图进行显示。
    #  将acc打印出来。
    plt.subplot(1, 2, 1)  # 将图像分为一行两列，将其显示在第一列
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)  # 将其显示在第二列
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('./img/LSTM_acc_loss.png')
    plt.show()
    plt.clf()

# 计算常用指标
def compute_indexes(y_test, y_test_pred):
    print(classification_report(y_test, y_test_pred))
    #混淆矩阵 
    cm = confusion_matrix(y_test, y_test_pred)
    df=pd.DataFrame(cm,index=["0", "1"],columns=["0", "1"])
    sns.heatmap(df,annot=True,fmt="d")
    plt.savefig('./img/LSTM_confusion_matrix.png')
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