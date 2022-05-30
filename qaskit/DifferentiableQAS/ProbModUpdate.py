import numpy as np
from QuantumProcess import *
from Basic import Basic
from Method import Method


def sofsum(alpha):
    sum1 = 0
    for i in range(len(alpha)):
        for j in range(len(alpha[i])):
            sum1 = sum1 + np.exp(alpha[i][j])
    return sum1

def softmax(j, alphai,sum1):
    # softmax categorical probability
    return np.exp(alphai[j]) / sum1


def cal_pro(alpha, id_listjm,sum1):
    prod = 1
    for i in range(len(alpha)):
        prod=prod*softmax(id_listjm,alpha[i],sum1)
    return prod


def delta(j, m):
    if abs(j - m) <= 1e-13:
        a = 1
    else:
        a = 0
    return a


def gradient(alpha, id_lists, costs):
    """
    alpha[p][c]:p layer_num, c op_num
    id_list[layer]: id_list[i]<=c-1
    cost[i]:
    """
    grad = [[0.0 for _ in range(len(alpha[0]))] for i in range(len(alpha))]
    sum1=sofsum(alpha)

    for i in range(len(alpha)):
        for j in range(len(alpha[i])):
            for k in range(len(id_lists)):
                ln = -cal_pro(alpha, id_lists[k][i],sum1) + delta(id_lists[k][i], j)
                grad[i][j] += ln * costs[k]

    '''
    for i in range(len(alpha)):
        for j in range(len(alpha[i])): 
            alphanew[i][j]+=grad[i][j] 
    '''
    return grad


class ProbModUpdate(Method):
    def __init__(self, basic: Basic):
        super().__init__(basic)

    def __call__(self, alpha, id_lists, costs, learning_rate):
        n_operations = len(alpha[0])
        new_alpha = [[0.0 for _ in range(n_operations)] for i in range(self.n_layers)]
        grad_alpha = gradient(alpha, id_lists, costs)
        for p in range(self.n_layers):
            for c in range(n_operations):
                new_alpha[p][c] = alpha[p][c] - learning_rate * grad_alpha[p][c]
        return new_alpha
