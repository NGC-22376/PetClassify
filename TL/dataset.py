"""
定义数据预处理，从原始图片得到网络的输入数据
"""

from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms


def data_process(img):
    # 获取图像输入尺寸
    width, height = img.size
    # 裁切
    center_crop = transforms.CenterCrop((height * 90, width * 90))
    # 归一化(利用ImageNet上图片的均值和方差)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # resize图像分辨率为(384, 384)
    resize = transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC)
    # 转成Tensor格式
    to_tenser = transforms.ToTensor()
    # 整合
    transform = transforms.Compose(
        [
            center_crop,
            resize,
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
