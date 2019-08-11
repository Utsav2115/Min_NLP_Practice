# CwsPosNerEntityRecognition
Chinese &amp; English Cws Pos Ner Entity Recognition implement using CNN bi-directional lstm and crf model with char embedding.基于字向量的CNN池化双向BiLSTM与CRF模型的网络，可能一体化的完成中文和英文分词，词性标注，实体识别。主要包括原始文本数据，数据转换,训练脚本,预训练模型,可用于序列标注研究.注意：唯一需要实现的逻辑是将用户数据转化为序列模型。分词准确率约为93%，词性标注准确率约为90%，实体标注（在本样本上）约为85%。  
# tips
中文分词，词性标注，实体识别，在使用上述模型时，本质是就是标注问题！！！  
如果你第一次使用相关的模型，只需要将 self.class_dict 里的词典改为你所需要的词典，然后将你的文本数据转换成列向量（竖着的数据形式，第三步的样子即可）。文本数据各种各样，对于很多第一次入门的小伙伴来说，最难的部分反而是数据转换。运行成功后，可以对参数，超参数，网络结构进行调参即可。  
主要实现使用了基于字向量的四层双向LSTM与CRF模型的网络.该项目提供了原始训练数据样本与转换版本,训练脚本,预训练模型,可用于序列标注研究.把玩和PK使用.  

# 项目介绍
医学实体识别是让计算机理解病历、应用病历的基础。基于对病历的结构化，可以计算出症状、疾病、药品、检查检验等多个知识点之间的关系及其概率，构建医疗领域的知识图谱，进一步优化医生的工作.
CCKS2018的电子病历命名实体识别的评测任务，是对于给定的一组电子病历纯文本文档，识别并抽取出其中与医学临床相关的实体，并将它们归类到预先定义好的类别中。组委会针对这个评测任务，提供了600份标注好的电子病历文本，共需识别含解剖部位、独立症状、症状描述、手术和药物五类实体。
领域命名实体识别问题自然语言处理中经典的序列标注问题, 本项目是运用深度学习方法进行命名实体识别的一个尝试.

# 实验数据
一, 目标序列标记集合
    O非实体部分,TREATMENT治疗方式, BODY身体部位, SIGN疾病症状, CHECK医学检查, DISEASE疾病实体,
二, 序列标记方法
    采用BIO三元标记


    self.class_dict ={
                     'O':0,
                     'TREATMENT-I': 1,
                     'TREATMENT-B': 2,
                     'BODY-B': 3,
                     'BODY-I': 4,
                     'SIGNS-I': 5,
                     'SIGNS-B': 6,
                     'CHECK-B': 7,
                     'CHECK-I': 8,
                     'DISEASE-I': 9,
                     'DISEASE-B': 10
                    }

三, 数据转换  

模型输出样式:

    ，	O
    男	O
    ，	O
    双	O
    塔	O
    山	O
    人	O
    ，	O
    主	O
    因	O
    咳	SIGNS-B
    嗽	SIGNS-I
    、	O
    少	SIGNS-B
    痰	SIGNS-I
    1	O
    个	O
    月	O
    ，	O
    加	O
    重	O
    3	O
    天	O
    ，	O
    抽	SIGNS-B
    搐	SIGNS-I


# 模型搭建
   本模型使用预训练字向量,作为embedding层输入,然后经过两个双向LSTM层进行编码,编码后加入dense层,最后送入CRF层进行序列标注.

       '''使用预训练向量进行模型训练'''
    def tokenvec_bilstm2_crf_model(self):
        model = Sequential()
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=True)
        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(self.NUM_CLASSES)))
        crf_layer = CRF(self.NUM_CLASSES, sparse_target=True)
        model.add(crf_layer)
        model.compile('adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
        model.summary()
        return model


# 模型效果  
1, 模型的训练:  

   | 模型 | 训练集 | 测试集 |训练集准确率 |测试集准确率 |备注|
   | :--- | :---: | :---: | :--- |:--- |:--- |
   | 医疗实体识别 | 6268 | 1571| 0.9649|0.8451|5个epcho|


模型  
![accuracy](https://github.com/liweimin1996/CwsPosNerCNNRNNLSTM/blob/master/accuracy.png)  

![loss](https://github.com/liweimin1996/CwsPosNerCNNRNNLSTM/blob/master/loss.png)  

2, 模型的测试:
    python lstm_predict.py, 对训练好的实体识别模型进行测试,测试效果如下:

        enter an sent:他最近头痛,流鼻涕,估计是发烧了
        [('他', 'O'), ('最', 'O'), ('近', 'O'), ('头', 'SIGNS-B'), ('痛', 'SIGNS-I'), (',', 'O'), ('流', 'O'), ('鼻', 'O'), ('涕', 'O'), (',', 'O'), ('估', 'O'), ('计', 'O'), ('是', 'O'), ('发', 'SIGNS-B'), ('烧', 'SIGNS-I'), ('了', 'SIGNS-I')]
        enter an sent:口腔溃疡可能需要多吃维生素
        [('口', 'BODY-B'), ('腔', 'BODY-I'), ('溃', 'O'), ('疡', 'O'), ('可', 'O'), ('能', 'O'), ('需', 'O'), ('要', 'O'), ('多', 'O'), ('吃', 'O'), ('维', 'CHECK-B'), ('生', 'CHECK-B'), ('素', 'TREATMENT-I')]
        enter an sent:他骨折了,可能需要拍片
        [('他', 'O'), ('骨', 'SIGNS-B'), ('折', 'SIGNS-I'), ('了', 'O'), (',', 'O'), ('可', 'O'), ('能', 'O'), ('需', 'O'), ('要', 'O'), ('拍', 'O'), ('片', 'CHECK-I')]  


# 总结    
1,本项目针对医学命名实体任务,实现了一个基于Bilstm+CRF的命名实体识别模型  
2,本项目使用charembedding作为原始特征,训练集准确率为0.9649,测试集准确达到0.8451  
3,命名实体识别可以加入更多的特征进行训练,后期将逐步实验其他方式.  




参考：
隐马尔科夫模型（HMM）一基本模型与三个基本问题 - 忆臻的文章 - 知乎  
https://zhuanlan.zhihu.com/p/26811689  

https://github.com/liuhuanyong/MedicalNamedEntityRecognition  