"""
定义用于训练的函数
"""
import torch
import torch.nn as nn

from TL.utils import output_msg_with_time, draw, save_checkpoint
from torch.utils.data import DataLoader
from extract_features import extract_features
from model import model


def init_weight(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.xavier_uniform_(layer.weight)


def evaluate(net, test_data, batch_size, loss, device):
    """
    :return: 测试的平均误差
    """
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    total_loss = 0
    net.eval()
    model.eval()
    # 测试
    acc = 0
    times = 0
    for X, y in test_dataloader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            # features = extract_features(X, model, batch_size, device=device)
            # x = torch.stack([feature for feature in features], dim=0)
            y_hat = net(X)
            eval_loss = loss(y_hat, y)

        # 计算每一批量的准确个数，并累加
        equal = torch.eq(y, torch.argmax(y_hat, dim=1))
        acc += torch.sum(equal, dim=0)
        # 累加所有批量的总误差
        total_loss = total_loss + eval_loss
        # 打印批量完成信息
        times += 1
        output_msg_with_time(f"test:{times * batch_size}")

    batch_num = len(test_dataloader)
    avg_test_loss = total_loss / (batch_num)
    avg_acc = acc / (batch_num * batch_size)
    output_msg_with_time(f"本轮次测试平均误差：{avg_test_loss}, 平均准确率：{avg_acc}")
    return avg_test_loss, avg_acc


def train(net, train_data, test_data, batch_size, epochs, device, lr=0.5):
    # 初始化网络参数
    net.apply(init_weight)

    # 转到指定设备
    try:
        net.to(device)
        print(f"模型设备: {next(net.parameters()).device}")
    except RuntimeError as e:
        print(f"模型转移到设备时出错: {e}")

    # 定义优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-4)
    loss = nn.CrossEntropyLoss()

    loss_list = [[], []]
    accuracy_list = [[], []]
    # 开始训练
    for epoch in range(1, epochs + 1):
        times = 0
        output_msg_with_time(f"训练轮次：{epoch}")
        net.train()
        total_loss = 0
        accuracy = 0
        print("Loading data finished.")
        train_dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0, drop_last=True)
        for X, y in train_dataloader:
            # 每batch的训练全程
            optimizer.zero_grad()
            x, y = X.to(device), y.to(device)
            y_hat = net(x)
            train_loss = loss(y_hat, y)
            train_loss.backward()
            optimizer.step()

            # 计算每一批量的准确个数，并累加
            equal = torch.eq(y, torch.argmax(y_hat, dim=1))
            accuracy += torch.sum(equal, dim=0)

            # 所有测试batch的累计误差
            total_loss = total_loss + train_loss

            # 打印批量完成信息
            times += 1
            output_msg_with_time(f"train:{batch_size * times}")

        # 得到每个测试batch的平均误差
        batch_num = len(train_dataloader)
        avg_loss = total_loss / (batch_num)
        avg_acc = accuracy / (batch_num * batch_size)
        output_msg_with_time(f"本轮次训练平均误差：{avg_loss}, 平均准确率：{avg_acc}")
        loss_list[0].append(float(avg_loss))
        accuracy_list[0].append(avg_acc)

        # 保存checkpoint文件
        save_checkpoint(net=net, optimizer=optimizer, epoch=epoch, path="./checkpoint.pth")

        # 测试
        output_msg_with_time(f"开始第{epoch}轮次测试")
        test_loss, test_acc = evaluate(net, test_data, batch_size, loss, device)
        loss_list[1].append(test_loss)
        accuracy_list[1].append(test_acc)

        # 绘图
        draw(loss_list, accuracy_list, epoch, epochs)
