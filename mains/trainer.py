import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from utils.config import config
from tqdm import tqdm
from modules.dpcnn import DPCNN
import numpy as np
import mindspore.nn as nn
from data_process.dataloader import ModelDataProcessor
from gensim.models import FastText
from mindspore import Parameter, Tensor
import mindspore

from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class Trainer:
    def __init__(self, 
                train_data,
                test_data,
                embedding_pre=None
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.model = DPCNN(embedding_pre)

        self.data_processor = ModelDataProcessor()
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_processor.get_data_loader()


        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction='mean')
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=config.lr)

    
    def train(self):
        net_with_criterion = nn.WithLossCell(self.model, self.criterion)
        train_network = nn.TrainOneStepCell(net_with_criterion, self.optimizer)
        train_network.set_train()

        for epoch in range(config.epochs):
            for x_batch, y_batch in self.data_processor.get_batch(self.X_train, self.y_train):
                x_batch = Tensor(x_batch, mindspore.int32)
                y_batch = Tensor(y_batch, mindspore.int32)
                loss = train_network(x_batch, y_batch)

        