import timm
from dataset import MyDataset
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from model import model
from train import output_msg_with_time

# 获取数据
output_msg_with_time("开始获取训练数据")
dataset_dir = r'C:\Users\30744\Desktop\CodeFiles\Python\MyPetClassification\Dataset'
train_cat_dataset = MyDataset(dataset_dir + "\\train", "Cat")
train_dog_dataset = MyDataset(dataset_dir + "\\train", "Dog")
train_data = ConcatDataset([train_cat_dataset, train_dog_dataset])
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
output_msg_with_time("数据加载完毕")

# 获取训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取特征文件保存路径
save_path = '../train_features.npz'


# 保存日志，防止内存不足造成程序中断而导致的特征数据丢失
def save_log(log_path, image_count):
    with open(log_path, 'a') as log_file:
        log_file.write(f"提取的图片数量: {image_count}\n")
    print(f"图片数量已保存到日志文件: {log_path}")


def extract_features(img, net, device):
    print(device)
    img.to(device)
    net.to(device)
    backbone = model.eval()
    return backbone(img)[-1]


def save_features(net, save_path):
    """
    :param net: 提取特征的网络
    :param save_path: 保存特征文件的路径
    :return:
    """
    features = []
    labels = []

    output_msg_with_time("开始提取特征")
    for X, y in train_dataloader:
        try:
            feature = extract_features(X, net, device)
            features.append(feature.cpu().numpy())
            labels.append(y.numpy())
            print(f"extract feature: {len(labels)}")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # 记录提取的图片数量到日志文件
                save_log('../extract_log.txt', len(labels))
                # 手动清理缓存，防止占用内存过多
                torch.cuda.empty_cache()
                break

    output_msg_with_time("特征提取完成")

    # 转换所有特征为numpy数组
    all_features = np.concatenate(features, axis=0)
    all_labels = np.concatenate(labels, axis=0)

    # 保存特征
    np.savez(save_path, features=all_features, labels=all_labels)
    print(f"特征已保存到 {save_path}")


def load_saved_features(feature_file=save_path):
    data = np.load(feature_file)
    features = torch.tensor(data['features'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.long)
    return features, labels


if __name__ == '__main__':
    save_features(model, save_path)
