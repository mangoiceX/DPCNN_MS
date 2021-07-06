import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import mindspore
import mindspore.nn as nn
from utils.config import config
import mindspore.ops.operations as P
import mindspore.ops as ops


class DPCNN(nn.Cell):
    def __init__(self, embedding_pre=None):
        super().__init__()
        self.embedding_layer = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        si = 3
        self.region_embedding = nn.Conv2d(1, config.num_filter, (3, config.embedding_dim), stride=1, pad_mode='valid')
        self.conv = nn.Conv2d(config.num_filter, config.num_filter, (si, 1), stride=1, pad_mode='valid')
        self.act_fun = nn.ReLU()
        self.fc = nn.Dense(config.num_filter, config.num_rel)
        self.padding0 = nn.Pad(paddings=((0,0),(0,0),(1,1),(0,0)), mode="CONSTANT")  # 针对行，上下添加一行0
        self.padding1 = nn.Pad(paddings=((0,0),(0,0),(0,1),(0,0)), mode="CONSTANT")  # 下添加一行0 
        self.pooling = nn.MaxPool2d(kernel_size=(si, 1), stride=2)

    def construct(self, X):
        word_embeddings = self.embedding_layer(X)  # [batch_size, seq_len, embedding_dim]
        word_embeddings = P.ExpandDims()(word_embeddings, 1)  # [batch_size, 1, seq_len, embedding_dim]
        region_word_embeddings = self.region_embedding(word_embeddings)  # [batch_size, num_filter, seq_len-3+1, 1]

        x = self.padding0(region_word_embeddings)  # [batch_size, num_filter, seq_len, 1]
        x = self.conv(self.act_fun(region_word_embeddings))  # [batch_size, num_filter, seq_len-3+1, 1]
        x = self.padding0(x)  # [batch_size, num_filter, seq_len, 1]

        x = self.conv(self.act_fun(x))  # [batch_size, num_filter, seq_len-3+1, 1]
        x = self.padding0(x)
        x = x + region_word_embeddings  # 残差连接
        
        while x.shape[-2] >= 2:  # 直到的seq_len数量减少到1
            x = self._block(x)
            # x = self.batch_normer2(x)
        x = ops.Squeeze()(x)  # [batch_size, num_filter, 1, 1] -> [batch_size, num_filters]
        # print(x.shape)
        # x = x.view(config.batch_size, config.num_filter)  # [batch_size, num_class_num]
        x = self.fc(x)

        return x

    def _block(self, x):

        x = self.padding1(x)
        px = self.pooling(x)  # [batch_size, (seq_len-2-2)/2, 1, num_filter]

        # 下面是两个等长卷积模块
        x = self.padding0(px) # 
        x = self.conv(self.act_fun(x))
        
        x = self.padding0(x)
        # print(x)
        x = self.conv(self.act_fun(x))

        # 残差连接
        x = x + px
    
        return x
        
