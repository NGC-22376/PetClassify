"""
网络的默认参数，在训练和测试中用到
"""
from easydict import EasyDict


def set_config(args):
    if not args.run_distribute:
        args.run_distribute = False
    config_cpu = EasyDict({
        # 类别数量
        "num_classes": 2,
        "image_height": 256,
        "image_width": 256,
        "batch_size": 32,
        # epoch指使用整个训练数据集的一次完整迭代，epoch_size则是在一个完整迭代的过程中需要遍历的样本数量或迭代次数
        "epoch_size": 30,
        # 学习率预热，指训练初期采用较小学习率，并逐渐增大学习率的过程
        "warmup_epochs": 0,
        "lr_init": .0,
        # 调整模型在训练末期的学习行为，以提高模型的瘦脸效果和泛化能力
        "lr_end": 0.01,
        # lr_max < lr_end 通常是因为采用动态调整学习率的策略，如余弦退火学习率调度器
        "lr_max": 0.001,
        # 过去梯度对当前参数更新的影响程度
        "momentum": 0.9,
        # 通过在损失函数中添加一个正则化项来减小权重的大小，以防止过拟合
        "weight_decay": 4e-5,
        # 设置平滑处理参数。标签平滑，一种正则化技术，用于减少模型对训练数据中噪声的过度拟合
        "label_smooth": 0.1,
        # 损失缩放因子，用于放大损失函数的计算结果，避免梯度值过小带来的问题
        "loss_scale": 1024,
        # 保存检查点文件，便于后续对模型的恢复
        "save_checkpoint": True,
        # 每隔多少个训练过程保存一次检查点文件
        "save_checkpoint_epochs": 1,
        "keep_checkpoint_max": 20,
        "save_checkpoint_path": "./",
        "platform": args.platform,
        "run_distribute": args.run_distribute,
        # 激活函数的类型
        "activation": "Softmax",
        # 类别映射，将模型输出的类别标签映射回原始类别
        "map": ["cat", "dog"]
    })
    return config_cpu
