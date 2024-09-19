from dataset import MyDataset, NpzDataset
from model import MobileNetV4Classifier
from train import train
from torch.utils.data import ConcatDataset
import torch

if __name__ == '__main__':
    save_path = '../train_features.npz'
    # # 加载训练数据集（特征）
    # train_dataset = NpzDataset(save_path)

    # 生成两类别各自的测试数据集
    dataset_dir = r'C:\Users\30744\Desktop\CodeFiles\Python\MyPetClassification\Dataset'

    # 加载训练数据集（一般）
    train_cat_dataset = MyDataset(dataset_dir + "\\train", "Cat")
    train_dog_dataset = MyDataset(dataset_dir + "\\train", "Dog")

    test_cat_dataset = MyDataset(dataset_dir + "\\eval", "Cat")
    test_dog_dataset = MyDataset(dataset_dir + "\\eval", "Dog")

    # 拼接生成数据集
    train_dataset = ConcatDataset([train_cat_dataset, train_dog_dataset])
    test_dataset = ConcatDataset([test_cat_dataset, test_dog_dataset])

    model = MobileNetV4Classifier(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(net=model, train_data=train_dataset, test_data=test_dataset, batch_size=64, epochs=8, device=device, lr=0.1)
