import sys 
from collections import defaultdict
import jieba
import gensim
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os
from config import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 建立一个字典(词-索引的映射), 一个表(索引-词向量的矩阵)
def build_embeddings_matrix(word_vec_model):
    # 初始化词索引字典
    word_index = defaultdict(dict)
    # 初始化词向量矩阵
    embeddings_matrix = np.zeros((len(word_vec_model.wv.key_to_index)+1, 128))
    # 填写词引索字典和词向量表
    for index, word in enumerate(word_vec_model.wv.index_to_key):
        word_index[word] = index + 1
        embeddings_matrix[index+1] = word_vec_model.wv.get_vector(word)
    return word_index, embeddings_matrix

# 生成三组数据集(训练集, 验证集, 测试集)(针对深度学习模型)
def get_data_LSTM(word_index):
    df = pd.read_csv(waimai_cut_path, names=["label", "review"])
    # 提取向量
    df["word_vector"] = df["review"].astype("str").map(lambda x: np.array([word_index.get(i, 0) for i in x.split(" ")]))
    # 填充及截断,限制每句话为50个分词字符串
    train = keras.preprocessing.sequence.pad_sequences(df["word_vector"].values, maxlen=50, padding='post', truncating='post', dtype="float32")
    # 从训练集上分出测试集
    x_train, x_test, y_train, y_test = train_test_split(train, df["label"].values, test_size=0.2)#, random_state=1
    # 从训练集上分出验证集
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
    return x_train, x_val, x_test, y_train, y_val, y_test

# 生成数据集(训练集, 测试集)(针对DTC模型的word2vec)
def get_data_DTC(word_index):
    df = pd.read_csv(waimai_cut_path, names=["label", "review"])
    # 提取向量
    df["word_vector"] = df["review"].astype("str").map(lambda x: np.array([word_index.get(i, 0) for i in x.split(" ")]))
    # 填充及截断,限制每句话为50个分词字符串
    train = keras.preprocessing.sequence.pad_sequences(df["word_vector"].values, maxlen=50, padding='post', truncating='post', dtype="float32")
    # 从训练集上分出测试集
    x_train, x_test, y_train, y_test = train_test_split(train, df["label"].values, test_size=0.2)#, random_state=1
    return x_train, x_test, y_train,  y_test