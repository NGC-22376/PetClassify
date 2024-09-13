"""
定义数据预处理，从原始图片得到网络的输入数据
"""
import numpy as np
import timm
from model import model
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
from TL.utils import load_saved_features


def data_process(img):
    # 获取图像输入尺寸
    width, height = img.size
    # 裁切
    center_crop = transforms.CenterCrop((256, 256))
    # 归一化(利用ImageNet上图片的均值和方差)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # resize图像分辨率为(256,256)
    resize = transforms.Resize((269, 269), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
    # 转成Tensor格式
    to_tenser = transforms.ToTensor()
    # 整合
    transform = transforms.Compose(
        [
            resize,
            center_crop,
            to_tenser,
            normalize
        ]
    )
    return transform(img)


class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        super().__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.dir_path = os.path.join(root_dir, label_dir)
        self.img_names = os.listdir(self.dir_path)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_path = os.path.join(self.dir_path, self.img_names[item])
        img = Image.open(img_path)
        img = data_process(img)
        # 标签数字化，猫为0，狗为1
        if self.label_dir == "Cat":
            label = 0
        else:
            label = 1
        return img, label


class NpzDataset(Dataset):
    def __init__(self, npz_path):
        self.data = np.load(npz_path)

    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, item):
        features, labels = load_saved_features(self.data)
        return features[item], labels[item]
