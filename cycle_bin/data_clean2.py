import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from utils.config import config

def read_data(filename):
    """
    数据读取
    :param filename: 文件路径
    :return: 数据读取内容（整个文档的字符串）
    """
    with open(filename, "r", encoding="utf8") as reader:
        content = reader.read()
    return content

def get_dict():
    '''
    将频数高的单词组成字典
    '''
    # '../data/rt-polaritydata/pos.txt'   '../data/rt-polaritydata/neg.txt'
    pos_text, neg_text = read_data('../data/rt-polaritydata/rt-polarity_processed.pos'), read_data('../data/rt-polaritydata/rt-polarity_processed.neg')
    total_text = pos_text + '\n' + neg_text
    text = total_text.split()

    # 分词
    vocab = [w for w, f in Counter(text).most_common() if f > 1]
    vocab = ['<pad>', '<unk>'] + vocab

    index2word = {i: word for i, word in enumerate(vocab)}

    word2index = {word: i for i, word in enumerate(vocab)}

    return len(word2index), index2word, word2index

def convert_text2index(sentence, word2index):  # , max_length=config.LARGE_SENTENCE_SIZE
    """
    将语料转成数字化数据
    :param sentence: 单条文本
    :param word2index: 词语-索引的字典
    :param max_length: text_cnn需要的文本最大长度
    :return: 对语句进行截断和填充的数字化后的结果 [1,max_length]
    """
    unk_id = word2index['<unk>']
    pad_id = word2index['<pad>']
    # 对句子进行数字化转换，对于未在词典中出现过的词用unk的index填充
    indexes = [word2index.get(word, unk_id) for word in sentence.split()]
    # 截断和填充
    # if len(indexes) < max_length:
    #     indexes.extend([pad_id] * (max_length - len(indexes)))
    # else:
    #     indexes = indexes[:max_length]

    return indexes

def process_file(file_name:str):
    setences_list = []
    with open(file_name, 'r', encoding='Windows-1252') as f:
        for line in f:
            text = line.rstrip().split()
            setences_list.append(text)

    return setences_list

def process_data(file_name_pos, file_name_neg):
    setences_list_pos = process_file(file_name_pos)
    setences_list_neg = process_file(file_name_neg)

    # 添加标签
    setences_list = setences_list_pos + setences_list_neg
    
    labels = [1 for i in range(len(setences_list_pos))] + [0 for i in range(len(setences_list_neg))]
    
    # 制作数据集
    X_train, X_test, y_train, y_test = train_test_split(setences_list, labels, test_size=0.3, shuffle=True, random_state=0, stratify=labels)

    return X_train, X_test, y_train, y_test

def get_data_loader():

    X_train, X_test, y_train, y_test = process_data('../data/rt-polaritydata/rt-polarity_processed.pos', '../data/rt-polaritydata/rt-polarity_processed.neg')

    vocal_size, index2word, word2index = get_dict()
    # 中间应该还增加对文本的编码
    train_text_ids = [[word2index[word] if word in word2index else 1 for word in item] for item in X_train]
    test_text_ids = [[word2index[word] if word in word2index else 1 for word in item] for item in X_test]

    return train_text_ids, test_text_ids, y_train, y_test

def get_batch(x, y):
    assert len(x) == len(y) , "error shape!"

    n_batches = int(len(x) / config.batch_size)  # 统计共几个完整的batch
    for i in range(n_batches - 1):
        x_batch = x[i*config.batch_size: (i + 1)*config.batch_size]
        y_batch = y[i*config.batch_size: (i + 1)*config.batch_size]
        lengths = [len(seq) for seq in x_batch]
        max_length = max(lengths)
        for i in range(len(x_batch)):
            x_batch[i] = x_batch[i] + [0 for j in range(max_length-len(x_batch[i]))]

        yield x_batch, y_batch

if __name__ == '__main__':
    vocal_size, index2word, word2index = get_dict()
    with open('../data/vocab2.txt', 'w') as f:
        for key in word2index:
            f.write(key)
            f.write('\n')
    
