"""
创建训练集和测试集
"""
import multiprocessing
import os

import mindspore.common.dtype as ms_type
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.c_transforms as dc
import mindspore.dataset.vision.c_transforms as vc
import numpy as np
from mindspore import Tensor
from mindspore.train.model import Model


def create_dataset(dataset_path, do_train, config, repeat_num=1):
    """
        create a train or eval dataset

        Args:
            dataset_path(string): the path of dataset.
            do_train(bool): whether dataset is used for train or eval.
            config(struct): the config of train and eval in different platform.
            repeat_num(int): the repeat times of dataset. Default: 1.

        Returns:
            dataset
        """
    # 获取可用的CPU核心数量，并限制在一定范围内：[1,8]
    cores = max(min(multiprocessing.cpu_count(), 8), 1)
    # 加载文件夹数据集，设定并行线程数，确定是否随机打乱
    ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=cores, shuffle=True)

    resize_height = config.image_height
    resize_width = config.image_width
    buffer_size = 1000

    # 图像数据增强：解码（像素数据-张量/数组） + 随机裁剪-解码-调整大小 + 一定概率进行随机水平反转
    decode_op = vc.Decode()
    resize_crop_op = vc.RandomCropDecodeResize(resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = vc.RandomHorizontalFlip(prob=0.5)

    # 调整图像到指定大小 + 随即调整图像的亮度、对比度和饱和度 + 像素值归一化处理（均值-0 标准差-1） + 更改图像的通道顺序
    resize_op = vc.Resize((resize_height, resize_width))
    rescale_op = vc.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    normalize_op = vc.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    change_swap_op = vc.HWC2CHW()

    if do_train:
        batch_size = config.batch_size
    else:
        batch_size = 1
    trans = [decode_op, resize_op, normalize_op, change_swap_op]
    type_cast_op = dc.TypeCast(ms_type.int32)

    # 对图像数据和标签数据分别处理
    ds = ds.map(operations=trans, input_columns="image", num_parallel_workers=cores)
    ds = ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=cores)

    # 数据洗牌
    ds = ds.shuffle(buffer_size=buffer_size)

    # 分组
    ds = ds.batch(batch_size, drop_remainder=True)

    # 重复遍历
    ds = ds.repeat(repeat_num)

    return ds


def extract_features(net, dataset_path, config):
    print("start cache feature!")
    features_folder = os.path.join(dataset_path, "features")

    train_dataset = create_dataset(dataset_path=os.path.join(dataset_path, "train"), do_train=True, config=config)
    eval_dataset = create_dataset(dataset_path=os.path.join(dataset_path, "eval"), do_train=False, config=config)

    train_size = train_dataset.get_dataset_size()
    eval_size = eval_dataset.get_dataset_size()
    if train_dataset.get_dataset_size() == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of train dataset is more \
            than batch_size in config.py")
    if os.path.exists(features_folder):
        train_features = np.load(os.path.join(features_folder, f"train_feature.npy"))
        train_labels = np.load(os.path.join(features_folder, f"train_label.npy"))
        eval_features = np.load(os.path.join(features_folder, f"eval_feature.npy"))
        eval_labels = np.load(os.path.join(features_folder, f"eval_label.npy"))
        return (train_features, train_labels, eval_features, eval_labels), train_size

    os.mkdir(features_folder)
    model = Model(net)
    train_feature = []
    train_labels = []
    eval_feature = []
    eval_labels = []

    extract_features_new(model, train_dataset, train_feature, f"train_feature", train_labels, f"train_label",
                         features_folder, config, train_size)
    extract_features_new(model, eval_dataset, eval_feature, f"eval_feature", eval_labels, f"eval_label",
                         features_folder, config, eval_size)

    return (np.array(train_feature), np.array(train_labels), np.array(eval_feature), np.array(eval_labels)), train_size


def extract_features_new(model, dataset, features, save_feature, labels, save_label, folder, config, size):
    images = size * config.batch_size
    for i, data in enumerate(dataset.create_dict_iterator()):
        # 用字典的形式把单个数据的图像信息和标签信息提取出来
        image = data["image"]
        label = data["label"]
        # 将图像数据输入到网络模型中进行预测，并保存预测结果于feature，通常是特征向量/分类结果
        feature = model.predict(Tensor(image))
        features.append(feature.asnumpy())
        labels.append(label.asnumpy())
        percent = round(i / size * 100., 2)
        # 即时输出特征提取信息
        print(f'{save_feature} cached [{i * config.batch_size}/{images}] {str(percent)}% ', end='\r', flush=True)
    np.save(os.path.join(folder, save_feature), np.array(features))
    np.save(os.path.join(folder, save_label), np.array(labels))
    print(f'{save_feature} cached [{images}/{images}] 100%  \n{save_feature} cache finished!', flush=True)
