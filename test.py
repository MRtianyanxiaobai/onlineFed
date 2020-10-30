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

# model = DNN()
# grads = []
# for param in model.parameters():
#     if param.grad is None:
#         grads.append(torch.zeros_like(param.data))
#     else:
#         grads.append(param.grad.data)

# for local, glob in zip(grads, model.parameters()):
#     glob.data = local

# for param in model.parameters():
#     print(param.data)

x = [1,2,3,4]
y = (index, i) for index, i in x if i < 3
print(y)