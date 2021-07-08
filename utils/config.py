import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from utils.arg_config import args

class Config:

    def __init__(self,
            embedding_dim=300,
        ):
        
        self.embedding_dim = embedding_dim
        
        self.num_filter = 32
        self.dropout_embedding_rate = 0.5
        self.num_rel = 2
        self.batch_size = 128
        # self.vocab_file = '../data/vocab_pretrain.txt'
        self.vocab_file = '../data/vocab.txt'

        cnt = 1  # 添加padd的位置
        with open(self.vocab_file, 'r') as f:
            for line in f:
                cnt += 1
        # cnt = 10828
        self.vocab_size = cnt
        self.epochs = 100
        self.lr = 1e-3
        
        self.using_pretrained_embedding = args['use_pre_embed']


config = Config()

