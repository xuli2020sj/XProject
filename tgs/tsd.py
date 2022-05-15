from matplotlib.pyplot import *
import numpy as np


def file2array(path, delimiter=' '):  # delimiter是数据分隔符
    fp = open(path, 'r', encoding='utf-8')
    string = fp.read()  # string是一行字符串，该字符串包含文件所有内容
    fp.close()
    row_list = string.splitlines()  # splitlines默认参数是‘\n’
    data_list = [[float(i) for i in row.strip().split(delimiter)] for row in row_list]
    return np.array(data_list)


def file_MLEM(a, b, x):  # 迭代法获得衰减矩阵
    e = 2
    while e > 0.001:  # 误差向量二范数的最大值，这个不同值会有不同的效果
        P = np.dot(a, x)  # 矩阵相乘
        ratio = np.divide(b, P)  # 矩阵数除，投影值与估计值之间的比值
        d = np.ones((1, 96))  # 创建1×96矩阵，注意是两个括号
        A1 = np.transpose(a)   # 将a矩阵转置
        zi = np.dot(A1, ratio)
        mu = np.transpose(np.dot(d, a))
        c = np.divide(zi, mu)  # 确定修正因子
        Xf = x
        x = np.multiply(x, c)   # 矩阵数乘
        e = np.linalg.norm(Xf - x)   # 默认计算向量的二范数
    return x


def matrix_block(t):  # 将矩阵分成24×4的块
    p = int(t / 24)
    q = int(t % 24)
    return p, q


A = file2array('T_A.txt')
B = file2array('T_B.txt')
X = np.ones((96, 1))
X = file_MLEM(A, B, X)

# 绘制扇形矩阵图像
Y = np.zeros((4, 24))
for i in range(96):
    m, n = matrix_block(i)
    Y[m, n] = X[i, 0]

imshow(Y, interpolation='nearest', cmap='bone', origin='lower')
colorbar(shrink=.92)
show()