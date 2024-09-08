from dataset import MyDataset
from model import MobileNetV4Classifier
from train import train
from torch.utils.data import ConcatDataset
import torch

if __name__ == '__main__':
    # 生成两类别各自的训练和测试数据集
    dataset_dir = r'C:\Users\30744\Desktop\CodeFiles\Python\MyPetClassification\Dataset'
    train_cat_dataset = MyDataset(dataset_dir + "\\train", "Cat")
    train_dog_dataset = MyDataset(dataset_dir + "\\train", "Dog")
    test_cat_dataset = MyDataset(dataset_dir + "\\eval", "Cat")
    test_dog_dataset = MyDataset(dataset_dir + "\\eval", "Dog")

    # 拼接生成训练数据集和测试数据集
    train_dataset = ConcatDataset([train_cat_dataset, train_dog_dataset])
    test_dataset = ConcatDataset([test_cat_dataset, test_dog_dataset])

    model = MobileNetV4Classifier(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(net=model, train_data=train_dataset, test_data=test_dataset, epochs=4, device=device)
