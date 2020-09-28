---
layout: post
title: kashgari中文命名实体识别   
date: 2020-9-28 
categories: blog
tags: [kaggle,机器学习命名实体识别]
description: 文章金句。
---

# kashgari中文命名实体识别

## 1.安装

`pip install kashgari==2.0.0`

会自动安装tensorflow最新版本（2.3.0）

## 2.功能

### 2.1 数据导入

`import kashgari`

`from typing import Tuple`

`from typing import List`

`from kashgari.logger import logger`

`from kashgari import utils`

`from kashgari.corpus import ChineseDailyNerCorpus,DataReader`

`def load_data(cls,subset_name: str = 'train',shuffle: bool = True) -> Tuple[List[List[str]], List[List[str]]]:`

​    `corpus_path = 'path'`

​    `if subset_name == 'train':`

​        `file_path = os.path.join(corpus_path, 'example.train')`

​    `elif subset_name == 'test':`

​        `file_path = os.path.join(corpus_path, 'example.test')`

​    `else:`

​        `file_path = os.path.join(corpus_path, 'example.dev')`

​    `x_data, y_data = DataReader.read_conll_format_file(file_path)`

​    `if shuffle:`

​        `x_data, y_data = utils.unison_shuffled_copies(x_data, y_data)`

​    `logger.debug(f"loaded {len(x_data)} samples from {file_path}. Sample:\n"`

​                    `f"x[0]: {x_data[0]}\n"`

​                    `f"y[0]: {y_data[0]}")`

​    `return x_data, y_data`

`train_x, train_y = load_data('train')`

`valid_x, valid_y = load_data('valid')`

`test_x, test_y = load_data('test')`

输入需要导入的path即可



### 2.2embedding

#### 2.2.1 BertEmbedding

`classkashgari.embeddings.BertEmbedding(model_folder: str, **kwargs)`

适用于bert系列的word embedding 提供模型位置即可，如bert-chinese，ernie_tensorflow版本。

#### 2.2.2 TransformerEmbedding

`classkashgari.embeddings.TransformerEmbedding(vocab_path: str, config_path: str, checkpoint_path: str, model_type: str = 'bert', **kwargs)`

适合其他基于transformer的wordembedding模型，如哈工大的robert，华为的NEZHA，需一次输入*vocab.txt*，*config.json*，*model.ckpt-100000*的路径以及最后的模型名称。同样支持bert系embedding的导入

#### 2.2.3使用

`from kashgari.embeddings import WordEmbedding, BertEmbedding, TransformerEmbedding`

`bert_embed = BertEmbedding('<模型位置>')`

`model = BiLSTM_Model(bert_embed, sequence_length=100)`

`model.fit(train_x, train_y, valid_x, valid_y)`

### 2.3 labeling model

可选BiGRU_Model,BiGRU_CRF_Model,BiLSTM_Model,BiLSTM_CRF_Model,CNN_LSTM_Model

如：

`from kashgari.tasks.labeling import BiGRU_Model,BiGRU_CRF_Model,BiLSTM_Model,BiLSTM_CRF_Model,CNN_LSTM_Model`

`model = BiLSTM_Model()`

`model.fit(train_x, train_y, valid_x, valid_y)`

`print(hyper)`

`# {'layer_blstm': {'units': 128, 'return_sequences': True}, 'layer_dropout': {'rate': 0.4}, 'layer_time_distributed': {},layer_activation': {'activation': 'softmax'}}`

`hyper['layer_blstm']['units'] = 32`

`model = BiLSTM_Model(hyper_parameters=hyper)`

#### 2.3.2tensorbord 回调

`model = BiLSTMModel()`

`tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)`

`# 每一步都会自动print准确率，召回率和F1score`

`eval_callback = EvalCallBack(kash_model=model,valid_x=valid_x,valid_y=valid_y,step=5)`

`model.fit(train_x,train_y,valid_x,valid_y,batch_size=100,callbacks=[eval_callback, tf_board_callback])`

## 3.其他功能

可以用于文本翻译和带标签的文本分类

## 4.模型导入

`loaded_model = kashgari.utils.load_model(<ModulePath>)`

`s = input('请输入文本：')`

`a=list(s)`

`c=model.predict_entities([a])`