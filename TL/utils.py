from datetime import datetime

import torch
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
import numpy as np


# 输出指定信息和当前时间
def output_msg_with_time(msg):
    current = datetime.now()
    hour, minute, second = current.hour, current.minute, current.second
    print(f"{msg}\t当前时间{hour}:{minute}:{second}")


# 绘制loss-epoch图像
def draw(loss_list, acc_list, epoch):
    # 清除之前的图
    clear_output(wait=True)

    # 绘制误差变化图
    x = np.arange(1, epoch + 1)
    loss_train = torch.tensor(loss_list[0]).cpu().numpy()
    loss_test = torch.tensor(loss_list[1]).cpu().numpy()
    acc_train = torch.tensor(acc_list[0]).cpu().numpy()
    acc_test = torch.tensor(acc_list[1]).cpu().numpy()

    plt.plot(x, loss_train, linestyle='--', color='red', label='train_loss/200')
    plt.plot(x, loss_test, linestyle='--', color='blue', label='test_loss/200')
    plt.plot(x, acc_train, linestyle='-', color='red', label='train_acc')
    plt.plot(x, acc_test, linestyle='-', color='blue', label='test_acc')

    # 坐标轴：x[1, 2, 3, 4], y[0, 0.05, ..., 1.00]
    xticks = np.arange(1, 5)
    yticks = np.arange(0, 1.05, 0.05)

    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel('epoch')
    plt.ylabel('loss/acc')
    plt.title('diagram for TL')
    # 图例
    plt.legend()
    # 网格线
    plt.grid(True)
    plt.show()


# 保存日志，防止内存不足造成程序中断而导致的特征数据丢失
def extract_log(log_path, image_count):
    with open(log_path, 'a') as log_file:
        log_file.write(f"提取的图片数量: {image_count}\n")
    print(f"图片数量已保存到日志文件: {log_path}")


# 保存ckpt文件
def save_checkpoint(net, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


# 载入ckpt文件
def load_checkpoint(net, optimizer, file_path):
    checkpoint = torch.load(file_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {file_path}, starting from epoch {epoch}")
    return epoch


def load_saved_features(data):
    features = torch.tensor(data['features'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.long)
    return features, labels
