# 模型搭建
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# 构建模型
def LSTM(word_index, embeddings_matrix):

    model = keras.models.Sequential()

    #嵌入层
    model.add(keras.layers.Embedding(
        input_dim=len(word_index)+1,    #词汇表大小
        output_dim=128,                 #词语向量的维度
        weights=[embeddings_matrix],
        input_length=100,               #当该值为常量时，表示一个文本词序列的长度。
        trainable=False
    ))

    #LSTM层
    model.add(keras.layers.LSTM(
        units=64,               #输出空间的维度
        activation='tanh',      #激活函数
        return_sequences=True   #是否返回最后一个输出
    ))

    #全局池化层
    model.add(keras.layers.GlobalAveragePooling1D())

    #2个全连接层
    model.add(keras.layers.Dense(32, activation=tf.nn.relu))#units: 正整数，输出空间维度。
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    #编译
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),#优化器
        loss='binary_crossentropy',#损失函数
        metrics=['accuracy']#评估指标
    )
    #输出模型结构信息
    model.summary()
    return model