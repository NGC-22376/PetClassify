"""
定义用于训练的函数
"""
import torch
import torch.nn as nn
from model import extract_features, save_checkpoint
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from datetime import datetime
from IPython.display import clear_output


def init_weight(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.xavier_uniform_(layer.weight)


def get_current_time():
    current = datetime.now()
    hour, minute, second = current.hour, current.minute, current.second
    print(f"当前时间{hour}:{minute}:{second}")


def evaluate(net, test_data, loss, device):
    """
    :return: 测试的平均误差
    """
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=0)
    total_loss = 0
    net.eval()
    # 测试
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        x = extract_features(X, device=device)
        eval_loss = loss(net(x), y)
        total_loss = total_loss + eval_loss
        avg_test_loss = total_loss / len(test_dataloader)
        print("本轮次测试平均误差：", avg_test_loss)
    return avg_test_loss


def draw(loss_list, epochs):
    # 清除之前的图
    clear_output(wait=True)

    # 绘制误差变化图
    x = range(1, epochs + 1)
    y_train = loss_list[0]
    y_test = loss_list[1]

    plt.plot(x, y_train, 'b.-', label='训练误差')
    plt.plot(x, y_test, 'r.-', label='测试误差')
    plt.xlabel('训练轮次')
    plt.ylabel('误差')
    plt.title('训练与测试误差变化图')
    # 图例
    plt.legend()
    # 网格线
    plt.grid(True)
    plt.show()


def train(net, train_data, test_data, epochs, device, lr=0.1):
    # 初始化网络参数
    net.apply(init_weight)

    # 转到指定设备
    try:
        net.to(device)
        print(f"模型设备: {next(net.parameters()).device}")
    except RuntimeError as e:
        print(f"模型转移到设备时出错: {e}")

    # 定义优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    # 开始训练
    loss_list = [[], []]
    for epoch in range(0, epochs):
        print("训练轮次：", epoch + 1, end='\t')
        get_current_time()
        net.train()
        total_loss = 0
        train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
        print("Loading data finished.")
        for X, y in train_dataloader:
            # 每batch的训练全程
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            x = extract_features(X, device=device)
            print("feature extract finished.")
            y_hat = net(x)
            train_loss = loss(y_hat, y)
            train_loss.backward()
            optimizer.step()

            # 所有测试batch的累计误差
            total_loss = total_loss + train_loss

        # 得到每个测试batch的平均误差
        batch_num = len(train_dataloader)
        avg_loss = total_loss / batch_num
        print("本轮次训练平均误差：", avg_loss)
        loss_list[0].append(float(avg_loss))

        # 保存checkpoint文件
        save_checkpoint(model=net, optimizer=optimizer, epoch=epoch + 1, path=r"./checkpoint.pth")

        # 测试
        test_loss = evaluate(net, test_data, loss, device)
        loss_list[1].append(test_loss)

        # 绘图
        draw(loss_list, epochs)
