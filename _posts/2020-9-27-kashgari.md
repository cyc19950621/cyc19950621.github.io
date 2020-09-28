---
布局：发布
标题：kashgari中文命名实体识别   
日期：2020-9-28 
类别：博客
标签：[kaggle，机器学习命名实体识别]
说明：文章金句。
---

＃kashgari中文命名实体识别

## 1.安装

`pip install kashgari == 2.0.0`

会自动安装tensorflow最新版本（2.3.0）

## 2.功能

### 2.1数据导入

进口喀什gar里

来自输入import tuple

来自输入导入列表

来自kashgari.logger导入记录器

来自喀什gar里进口工具

从kashgari.corpus导入ChineseDailyNerCorpus，DataReader

`def load_data（cls，subset_name：str ='train'，shuffle：bool = True）-> Tuple [List [List [str]]，List [List [str]]]：`

``corpus_path ='路径'`

``如果subset_name =='火车'：`

`file_path = os.path.join（corpus_path，'example.train'）`

``ELIF子集名称=='测试'：`

`file_path = os.path.join（corpus_path，'example.test'）`

``其他：`

`file_path = os.path.join（corpus_path，'example.dev'）`

``x_data，y_data = DataReader.read_conll_format_file（file_path）`

如果洗牌：

``x_data，y_data = utils.unison_shuffled_copies（x_data，y_data）`

logger.debug（f“从{file_path}加载了{len（x_data）}个示例。Sample：\ n”`

``f''x [0]：{x_data [0]} \ n“`

``f''y [0]：{y_data [0]}“）`

返回x_data，y_data`

`train_x，train_y = load_data（'train'）`

`valid_x，valid_y = load_data（'valid'）`

`test_x，test_y = load_data（'test'）`

输入需要引入的path即可



### 2.2嵌入

#### 2.2.1 BertEmbedding

`classkashgari.embeddings.BertEmbedding（model_folder：str，** kwargs）`

适用于bert系列的单词嵌入提供模型位置即可，如bert-chinese，ernie_tensorflow版本。

#### 2.2.2 TransformerEmbedding

`classkashgari.embeddings.TransformerEmbedding（vocab_path：str，config_path：str，checkpoint_path：str，model_type：str ='bert'，** kwargs）`

适合其他基于transformer的单词嵌入模型，如哈工大的robert，华为的NEZHA，需一次输入* vocab.txt *，* config.json *，* model.ckpt-100000 *的路径以及最后一个模型名称。同样支持bert系embedding的引进

#### 2.2.3使用

从kashgari.embeddings导入WordEmbedding，BertEmbedding，TransformerEmbedding`

bert_embed = BertEmbedding（'<模型位置>'）

模型= BiLSTM_Model（bert_embed，sequence_length = 100）

model.fit（train_x，train_y，valid_x，valid_y）

### 2.3标签模型

可选BiGRU_Model，BiGRU_CRF_Model，BiLSTM_Model，BiLSTM_CRF_Model，CNN_LSTM_Model

如：

来自kashgari.tasks.labeling import BiGRU_Model，BiGRU_CRF_Model，BiLSTM_Model，BiLSTM_CRF_Model，CNN_LSTM_Model`

模型= BiLSTM_Model（）
model.fit（train_x，train_y，valid_x，valid_y）

#### 2.3.1更改超参

超级= BiLSTM_Model.default_hyper_parameters（）
`打印（超级）`

＃＃'layer_blstm'：{'units'：128，'return_sequences'：True}，'layer_dropout'：{'rate'：0.4}，'layer_time_distributed'：{}，'layer_activation'：{'activation'：' softmax'}}`

`hyper ['layer_blstm'] ['units'] = 32`
模型= BiLSTM_Model（hyper_parameters = hyper）

#### 2.3.2tensorbord或

模型= BLSTMModel（）

tf_board_callback = keras.callbacks.TensorBoard（log_dir ='。/ logs'，update_freq = 1000）`

`＃每一步都会自动打印准确率，召回率和F1score`

`eval_callback = EvalCallBack（kash_model = model，valid_x = valid_x，valid_y = valid_y，step = 5）`

`model.fit（train_x，train_y，valid_x，valid_y，batch_size = 100，callbacks = [eval_callback，tf_board_callback]）`

## 3.其他功能

可以用于文本翻译和带标签的文本分类

## 4.模型导入
`loaded_model = kashgari.utils.load_model（<ModulePath>）`
s = input（'请输入文本：'）
一个=列表
`c = model.predict_entities（[a]）`