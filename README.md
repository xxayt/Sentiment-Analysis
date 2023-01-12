

# Sentiment Analysis based on DecisionTreeClassifier & LSTM

- **数据集** [**(waimai_10k)**](https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/waimai_10k)：外卖平台用户评论的情感分类数据集【正面情感4000+；负面情感8000+】

- **配置说明**：此次我在AutoDL的线上环境平台的实验环境中，配置了TensorFlow2.5.0、Python3.8、Cuda11.2的Unumtu18.04镜像环境。

- **效果**：

  |              |  DTC   |                             LSTM                             |
  | :----------: | :----: | :----------------------------------------------------------: |
  | 特征提取方法 | TF-IDF |                           word2vec                           |
  |  分类器模型  | 决策树 | [LSTM模型](https://github.com/xxayt/Sentiment-Analysis/blob/main/model_LSTM.py) |
  |     效果     |  78%   |                             83%                              |

- **详细内容**：查看[实验报告Report](https://github.com/xxayt/Sentiment-Analysis/blob/main/实验报告Report.pdf)