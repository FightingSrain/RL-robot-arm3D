import numpy as np

# def sigmoid(x):
#
#     return 1/(1 + np.exp(-x))
def sigmoid(inx):
    if inx >= 0:      # 对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0 / (1 + np.exp(-inx))
    else:
        return np.exp(inx)/(1 + np.exp(inx))