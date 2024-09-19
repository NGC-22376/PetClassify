from dataset import MyDataset
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from model import model
from TL.utils import output_msg_with_time, extract_log


def extract_features(img, net, batch_size, device):
    img = img.to(device)
    net = net.to(device)
    imgs = torch.chunk(img, batch_size, 0)
    features = []
    with torch.no_grad():
        for i in imgs:
            features.append(net(i)[-2].reshape(1, -1, 8, 8))
    return [i for i in features]


def save_features(net, batch_size, save_path):
    """
    :param net: 提取特征的网络
    :param save_path: 保存特征文件的路径
    :return:
    """

    times = 0
    feature_list, label_list = [], []
    output_msg_with_time("开始提取特征")
    for X, y in train_dataloader:
        try:
            features = extract_features(X, net, batch_size, device)
            times += 1
            output_msg_with_time(f"extract feature: {times * batch_size}")

            # 特征存入数组
            for feature in features:
                feature_list.append(feature.cpu().numpy())
            for label in y:
                label_list.append(label.numpy())

        except RuntimeError as e:
            if 'out of memory' in str(e):
                extract_log('../extract_log.txt', {times * batch_size})
                # 手动清理缓存，防止占用内存过多
                torch.cuda.empty_cache()
                break

    np.savez(save_path, features=np.concatenate(feature_list, axis=0), labels=label_list)
    feature_list.clear()  # 清理内存
    label_list.clear()

    output_msg_with_time("特征提取完成")


if __name__ == '__main__':
    # 获取数据
    batch_size=64
    output_msg_with_time("开始获取训练数据")
    dataset_dir = r'C:\Users\30744\Desktop\CodeFiles\Python\MyPetClassification\Dataset'
    train_cat_dataset = MyDataset(dataset_dir + "\\train", "Cat")
    train_dog_dataset = MyDataset(dataset_dir + "\\train", "Dog")
    train_data = ConcatDataset([train_cat_dataset, train_dog_dataset])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    output_msg_with_time("数据加载完毕")

    # 获取训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取特征文件保存路径
    save_path = '../train_features.npz'

    # 获取提取特征的模型
    model = model.eval()

    save_features(model, batch_size, save_path)
