import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from Algorithms.models.model import *

# num_inputs = 2
# num_examples = 20
# true_w = [2, -3.4]
# true_b = 4.2
# features = torch.randn(num_examples, num_inputs,
#                        dtype=torch.float32)
# labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
#                        dtype=torch.float32)

# def data_iter(batch_size, features, labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)  # 样本的读取顺序是随机的
#     for i in range(0, num_examples, batch_size):
#         j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
#         yield  features.index_select(0, j), labels.index_select(0, j)

# batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break

# torch.manual_seed(0)

# flag = torch.rand(20)
# flag2 = torch.rand(20)
# users = range(20)

# selected = [users[index] for index, val in enumerate(flag) if val < 0.5]

# for index, val in enumerate(flag):
#     if val < 0.5:
#         print(index, val)

# print(selected)

model = Mclr_Logistic()
# alpha = torch.exp(torch.abs(list(model.parameters())[0].data))
# for index, val in enumerate(alpha):
#     sumCol = torch.sum(val)
#     alpha[index] = torch.div(val, sumCol.item())
# print(alpha)

# print(list(model.parameters())[0])
# for index, global_param in enumerate(model.parameters()):
#     if index == 0:
#         alpha = torch.div(features, sumFea.item())
#         print(alpha)
#         global_param.data = global_param.data.mul(alpha)
#         print(global_param.data)
torch.cuda.set_device(0)
# model = Net()
model = model.cuda()
while True:
    model.train()
