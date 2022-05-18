import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader

def show_result(train_loss,val_loss):
    x=list(range(0,len(train_loss)))
    plt.plot(x,train_loss)
    plt.plot(x, val_loss)
    plt.show()




class Mydata(Dataset):
    def __init__(self,x,y):
        super(Mydata,self).__init__()
        self.x=x
        self.y=y
    def __getitem__(self,index):
        x_sample=self.x[index]
        # x_sample=(x_sample-x_sample.mean())/x_sample.std()
        y_sample=self.y[index]

        return torch.Tensor(x_sample),torch.Tensor(y_sample)

    def __len__(self):
        return len(self.x)

class Mymodel(nn.Module):
    def __init__(self,length):
        super(Mymodel,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(length, 4 * length),
            nn.ReLU(),
            nn.Linear(4 * length, 8 * length),
            nn.ReLU(),
            nn.Linear(8 * length, 32 * length),
            nn.ReLU(),
            nn.Linear(32 * length,4 * length),
            nn.ReLU(),
            nn.Linear(4 * length,2 * length),
            nn.ReLU(),
            nn.Linear(2 * length, 1)
        )
    def forward(self,x):
        out=self.net(x)
        return out


def Get_loader(inputs,outputs):
    inputs = np.linspace(-10,10, 10000).reshape(10000,1)

    noise = np.random.normal(0, 0.1, (10000,1))
    inputs += noise
    outputs = inputs * inputs
    # print(inputs[0:10])
    # print(inputs.shape)
    # print(outputs.shape)

    batch_size = 8
    val_split = 0.2
    nums = len(inputs)

    x_train = inputs[int(val_split * nums):]
    y_train = outputs[int(val_split * nums):]
    x_val = inputs[0:int(val_split * nums)]
    y_val = outputs[0:int(val_split * nums)]

    train_data = Mydata(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_data = Mydata(x_val, y_val)
    val_loader = DataLoader(val_data, batch_size, shuffle=False)
    return train_loader,val_loader

def train(epochs,inputs,outputs):
    length=1
    model=Mymodel(length)
    lr=0.0001
    criterion=nn.MSELoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,weight_decay=0.001)
    train_loader,val_loader=Get_loader(inputs,outputs)
    total_train_loss=[]
    total_val_loss = []
    for epoch in range(epochs):
        train_loss=0.0
        val_loss = 0.0
        model.train()
        for idx ,(x,y)in enumerate(train_loader):
            optimizer.zero_grad
            out=model(x)
            loss=criterion(out,y)
            loss.backward()
            optimizer.step()
            train_loss+=loss
        train_loss/=len(train_loader)
        total_train_loss.append(train_loss)
        with torch.no_grad():
            model.eval()
            for idx, (x, y) in enumerate(val_loader):
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss
            val_loss /= len(val_loader)
            total_val_loss.append(val_loss)
        print(f"epoch:{epoch + 1:03d} train_loss:{train_loss:.5f} val_loss:{val_loss:.5f}")

if __name__ == '__main__':

    inputs = np.linspace(-3, 5, 1000).reshape(1000, 1)
    noise = np.random.normal(0, 0.01, (1000, 1))
    inputs += noise
    outputs = inputs * inputs

    train(1000,inputs,outputs)















