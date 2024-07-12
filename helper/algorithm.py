import torch
import torch
import torchvision

from torch.utils import data
from torchvision import transforms

import torch


def sgd(params, lr, batch_size) -> None:
    """⼩批量随机梯度下降
    :param params: 要更新的参数列表
    :param lr: 学习率
    :param batch_size: 批量大小
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def softmax(target) -> torch.Tensor:
    """softmax函数定义,将输入的向量指数概率化"""
    target_max = torch.max(target)
    target_exp = torch.exp(target - target_max)
    partition = target_exp.sum(1, keepdim=True)
    return target_exp / partition


def cross_entropy(y_hat, y) -> torch.Tensor:
    # 添加一个非常小的值以避免 log(0) 的情况
    eps = 1e-9
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
