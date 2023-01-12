# -*- coding: utf-8 -*-
import jieba
import gensim
import pandas as pd

waimai_10k_path = "DATA/waimai_10k/waimai_10k.csv"  # VScode路径
stopwords_path = "DATA/waimai_10k/baidu_stopwords.txt"
waimai_cut_path = "DATA/waimai_10k/waimai_cut.csv"


# 处理停用词
def stop_words(path):
    with open(path,'r',encoding='utf-8',errors='ignore') as file:
        return[x.strip() for x in file]

if __name__ == '__main__':
    stop_words = stop_words(path=stopwords_path)
    # 读取文件
    df = pd.read_csv(waimai_10k_path)
    print(df)
    # 切词并过滤调停用词
    df["review"] = df["review"].map(
        # 匿名函数
        lambda x: " ".join([i for i in jieba.cut(x) if i not in stop_words])
    )
    print(df)
    # 保存处理好的文本
    df.to_csv(waimai_cut_path, index=False, header=False, columns=["label","review"])
