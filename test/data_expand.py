"""
数据增广
"""
import os
import torchvision.transforms as transforms
from PIL import Image


def extend_img(img):
    width, height = img.size

    # 垂直反转
    flip_horiz = transforms.RandomVerticalFlip(0.8)

    # 随机裁剪
    random_crop = transforms.RandomResizedCrop((int(width * 0.5), int(height * 0.5)), (0.2, 1), (0.5, 2))

    # 改变图像亮度
    brightness_trans = transforms.ColorJitter(brightness=(0.2, 0.8))

    # 改变图像对比度
    contrast_trans = transforms.ColorJitter(contrast=(0.2, 0.8))

    return flip_horiz(brightness_trans(img)), random_crop(contrast_trans(img))


if __name__ == "__main__":
    data_root_dir = r"C:\Users\30744\Desktop\CodeFiles\Python\MyPetClassification\Dataset"
    dir = ["train"]
    species = ["Cat", "Dog"]

    for fir_dir in dir:
        for last_dir in species:
            img_dir = os.path.join(data_root_dir, fir_dir, last_dir)
            img_names = os.listdir(img_dir)
            for name in img_names:
                img = Image.open(os.path.join(img_dir, name))
                for index, i in enumerate(extend_img(img)):
                    i.save(os.path.join(img_dir, f'{index + 1}_{name}'))

