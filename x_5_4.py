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
    x = list(range(0, len(train_loss)))
    plt.plot(x, train_loss)
    plt.plot(x, val_loss)
    plt.show()


class Mydata(Dataset):
    def __init__(self, x, y):
        super(Mydata, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x_sample = self.x[index]
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
            nn.Linear(length, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 1000),
            nn.Mish(inplace=True),
            nn.Linear(1000, 1000),
            nn.Mish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1000, 1000),
            nn.Mish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1000, 1000),
            nn.Mish(inplace=True),
            nn.Dropout(0.2),
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


def train(epochs, inputs, outputs, batch_size, val_split, learn_rate, predict_df):
    length = inputs.shape[1]
    model = Mymodel(length)
    model = model.to(device)
    lr = learn_rate
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.002)

    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.1, last_epoch=-1)
    # base_lr = 0.001
    # max_lr =  0.005
    # lr_schedule = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2500, step_size_down=2500,
    #                                                 mode='triangular', gamma=0.1, scale_fn=None, scale_mode='cycle',
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
        total_train_loss.append(train_loss.detach().numpy())

        with torch.no_grad():
            model.eval()
            for idx, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.cpu()
            val_loss /= len(val_loader)
            total_val_loss.append(val_loss.detach().numpy())

            if epoch % 50 == 0:
                predict(model, predict_df)

        print(f"epoch:{epoch + 1:03d} train_loss:{train_loss:.12f} val_loss:{val_loss:.12f}")
    show_result(total_train_loss, total_val_loss)
    return model


def show(model, predict_df_):
    df = np.array(predict_df_)
    inputs = df[:, [2, 4]]

    acutal_out = df[:, 5].reshape(-1, 1)
    # plt.scatter(inputs[:[2]], acutal_out)

    y = [0.04652, 0.05954, 0.08803, 0.1221, 0.16585, 0.39169, 0.6617, 0.89806, 1.17324, 1.3325, 1.83608]

    inputs_tensor = torch.Tensor(inputs)
    model = model.to('cuda')
    model_out = model(inputs_tensor.to('cuda')).cpu().data.numpy()
    preision = (model_out - acutal_out) / acutal_out
    plt.scatter(inputs_tensor.data.numpy()[:, [0]], preision, color='red', linewidth=0.5)
    plt.scatter(inputs[:, [0]], acutal_out, color='y')
    plt.scatter(inputs[:, [0]], model_out, color='g')
    plt.show()


def predict(model, predict_df_):
    df = np.array(predict_df_)
    inputs = df[:, [2, 4]]

    acutal_out = df[:, 5].reshape(-1, 1)
    # plt.scatter(inputs[:[2]], acutal_out)

    # x1_pred = np.linspace(150, 650, 1001).reshape(1001, 1)
    # x2_pred = np.ones(1001) * 1.3325
    # x_pred = np.concatenate([x1_pred, x2_pred], axis=0)
    inputs_tensor = torch.Tensor(inputs)
    # x_pred=x
    # x1 =torch.Tensor(x)

    model = model.to('cuda')
    model_out = model(inputs_tensor.to('cuda')).cpu().data.numpy()
    preision = (model_out - acutal_out) / acutal_out
    plt.scatter(inputs_tensor.data.numpy()[:, [0]], preision, color='red', linewidth=0.5)
    plt.scatter(inputs[:, [0]], acutal_out, color='y')
    plt.scatter(inputs[:, [0]], model_out, color='g')
    plt.show()


if __name__ == '__main__':
    # 数据预处理
    df = pd.read_csv(r'C:\Users\X\PycharmProjects\XProject\data\data2.csv')
    # df['Efficiency'] = df['Efficiency'].map(lambda x: x * 1000)
    df = df.sample(frac=1)

    df = np.array(df)
    predict_df = df[:50]
    df = df[50:]
    inputs = df[:, [2, 4]]

    df_max = np.max(df[:, 2])
    df_min = np.min(df[:, 2])
    df[:, 2] = (df[:, 2] - df_min) / (df_max - df_min)

    batch_size = 48
    val_split = 0.05
    learn_rate = 0.001

    outputs = df[:, 5].reshape(-1, 1)

    model = train(1000, inputs, outputs, batch_size, val_split, learn_rate, predict_df)

    # predict(model,predict_df)
