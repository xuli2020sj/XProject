import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def show_result(train_loss, val_loss):
    print('show')
    # x = list(range(0, len(train_loss)))
    # plt.plot(x, train_loss)
    # plt.plot(x, val_loss)
    # plt.show()


class Mydata(Dataset):
    def __init__(self, x, y):
        super(Mydata, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x_sample = self.x[index]
        # x_sample=(x_sample-x_sample.mean())/x_sample.std()
        y_sample = self.y[index]
        return torch.Tensor(x_sample), torch.Tensor(y_sample)

    def __len__(self):
        return len(self.x)


class Mymodel(nn.Module):
    def __init__(self, length):
        super(Mymodel, self).__init__()
        in_node = length
        out_node = 1
        self.net = nn.Sequential(
            nn.Linear(length, 10),
            nn.Mish(inplace=True),
            nn.Linear(10, 100),
            nn.Mish(inplace=True),
            nn.Linear(100, 1000),
            nn.Mish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1000, 1000),
            nn.Mish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1000, 1000),
            nn.Mish(inplace=True),
            nn.Linear(1000, 1)
        )

    def forward(self, x):
        out = self.net(x)
        # out = self.fc(out)
        return out


def Get_loader(inputs, outputs, batch_size, val_split=0.2):
    nums = len(inputs)
    x_train = inputs[int(val_split * nums):]
    y_train = outputs[int(val_split * nums):]
    x_val = inputs[0:int(val_split * nums)]
    y_val = outputs[0:int(val_split * nums)]
    train_data = Mydata(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    val_data = Mydata(x_val, y_val)
    val_loader = DataLoader(val_data, batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader


def train(epochs, inputs, outputs, batch_size, val_split, learn_rate,predict_df):
    length = inputs.shape[1]
    model = Mymodel(length)
    model = model.to(device)
    lr = learn_rate
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.002)

    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.1, last_epoch=-1)
    # base_lr = 0.00005
    # max_lr =  0.0001
    # lr_schedule = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2500, step_size_down=2500,
    #                                                 mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
    #                                                 cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
    #                                                 last_epoch=-1)
    train_loader, val_loader = Get_loader(inputs, outputs, batch_size, val_split)
    total_train_loss = []
    total_val_loss = []
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            lr_schedule.step()
            train_loss += loss.cpu()
        train_loss /= len(train_loader)
        total_train_loss.append(train_loss)

        with torch.no_grad():
            model.eval()
            for idx, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.cpu()
            val_loss /= len(val_loader)
            total_val_loss.append(val_loss)
            if epoch % 50 == 0:
                predict(model, predict_df)

        print(f"epoch:{epoch + 1:03d} train_loss:{train_loss:.12f} val_loss:{val_loss:.12f}")
    show_result(total_train_loss, total_val_loss)
    return model


def predict(model,predict_df):
    df=predict_df
    df = np.array(df)
    inputs = df[:, 2:3]

    x = inputs
    y = df[:, 4].reshape(-1, 1)
    plt.scatter(x, y)

    x_pred = np.linspace(150, 650, 1001).reshape(1001, 1)
    x_pred = torch.Tensor(x_pred)
    # x_pred=x
    # x1 =torch.Tensor(x)
    x1=x_pred
    x1 = x1.to('cuda')
    model = model.to('cuda')
    out = model(x1)
    # print(out)
    out = out.cpu().data.numpy()
    plt.scatter(x_pred.data.numpy() ,out, color='red')
    # plt.plot(x, out, color='y')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\X\PycharmProjects\XProject\data\data2.csv')
    # df['Efficiency'] = df['Efficiency'].map(lambda x: x * 1000)
    df = df.sample(frac=1)
    batch_size = 32
    val_split = 0.2
    learn_rate = 0.001
    df = np.array(df)
    predict_df=df[:50]
    df=df[50:]
    inputs = df[:, 2:3]
    # mean_value = inputs.mean(axis=0)
    # std_value = inputs.std(axis=0)
    # inputs = (inputs - mean_value) / std_value
    outputs = df[:, 4].reshape(-1, 1)
    # outputs = [i * 1000 for i in outputs]

    model = train(2000, inputs, outputs, batch_size, val_split, learn_rate,predict_df)

    predict(model,predict_df)
