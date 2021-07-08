import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from utils.config import config
from tqdm import tqdm
from modules.dpcnn import DPCNN
import numpy as np
import mindspore.nn as nn
from data_process.data_loader import ModelDataProcessor
# from gensim.models import FastText
from mindspore import Parameter, Tensor
import mindspore


from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# from data_process.data_clean2 import get_data_loader, get_batch
class Trainer:
    def __init__(self, 
                embedding_pre=None
    ):
        self.model = DPCNN(embedding_pre)

        self.data_processor = ModelDataProcessor()
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_processor.get_data_loader()
        # self.X_train, self.X_test, self.y_train, self.y_test = get_data_loader()

        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction='sum')
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=config.lr)

    def controller(self):

        for epoch in range(config.epochs):
            print("Epoch: {}".format(epoch))
            self.train()
            self.test()

    def train(self):    
        net_with_criterion = nn.WithLossCell(self.model, self.criterion)
        train_network = nn.TrainOneStepCell(net_with_criterion, self.optimizer)
        train_network.set_train()

        loss_total = 0.0
        correct_total = 0
        for x_batch, y_batch in self.data_processor.get_batch(self.X_train, self.y_train):  # self.data_processor.
            x_batch = Tensor(x_batch, mindspore.int32)
            y_batch = Tensor(y_batch, mindspore.int32)
            # print(x_batch)
            loss = train_network(x_batch, y_batch)
            loss_total += float(loss.asnumpy())
            predict = self.model(x_batch).asnumpy().argmax(1)
            correct = 0
            for i, j in zip(predict, y_batch):
                correct += (i==j.asnumpy())
            correct_total += correct

        loss_total_final = loss_total
        accuracy = correct_total / len(self.X_train)
        print("train loss: {}, train accuracy: {}".format(loss_total_final, accuracy))

    def test(self):

        correct_total = 0
        for x_batch, y_batch in self.data_processor.get_batch(self.X_test, self.y_test):  # self.data_processor.
            x_batch = Tensor(x_batch, mindspore.int32)
            y_batch = Tensor(y_batch, mindspore.int32)
            predict = self.model(x_batch).asnumpy().argmax(1)
            correct = 0
            for i, j in zip(predict, y_batch):
                correct += (i==j.asnumpy())
            correct_total += correct

        accuracy = correct_total / len(self.X_test)
        print("test accuracy: {}".format(accuracy))

def get_embed():
    fname = '../data/embed_pre.txt'
    embedding_pre = []
    with open(fname, 'r') as f:
        for line in f:
            line = line[1:-2]
            line = line.split(',')
            # print(type(line))
            line = [float(num) for num in line]
            # print(line)
            embedding_pre.append(line)
    embedding_pre = np.array(embedding_pre)
    return embedding_pre


if __name__ == "__main__":
    # embedding_pre = get_embed()
    # embedding_pre = np.zeros((10827, 128))
    # embedding_pre = Tensor(embedding_pre, mindspore.float32)
    embedding_pre=None
    trainer = Trainer(embedding_pre)
    trainer.controller()

        