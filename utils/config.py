

from utils.arg_config import args

class Config:

    def __init__(self,
            embedding_dim=128,
        ):
        
        self.embedding_dim = embedding_dim
        
        self.num_filter = 8
        self.dropout_embedding_rate = 0.5
        self.num_rel = 2
        self.batch_size = 128
        self.vocab_file = '../data/vocab_pretrain.txt'

        cnt = 1  # 添加padd的位置
        with open(self.vocab_file, 'r') as f:
            for line in f:
                cnt += 1
        self.vocab_size = cnt
        self.epochs = 100
        self.lr = 1e-3
        
        self.using_pretrained_embedding = args['use_pre_embed']


config = Config()

