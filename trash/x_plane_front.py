import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader




if __name__ == '__main__':
    # 数据预处理
    df = pd.read_csv(r'C:\Users\X\PycharmProjects\XProject\data\data_9_plane_1.csv')
    # df['Efficiency'] = df['Efficiency'].map(lambda x: x * 100000)
    df = df.sample(frac=1)
    df = np.array(df)
    inputs = df[:, [2]]
    inputs -= 500
    outputs = df[:, 5].reshape(-1, 1)
    plt.yscale("log")
    plt.scatter(inputs[:, [0]], outputs, color='g')
    
    plt.show()
