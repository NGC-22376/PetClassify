"""
设置用于命令行调用的参数
"""

import argparse
import ast
import os


def train_parse_args():
    # 创建命令行解析器对象，定义命令行接口并解析用户提供的参数
    train_parser = argparse.ArgumentParser(description='Image classification train')
    train_parser.add_argument('--platform', type=str, default='CPU', choices=('CPU', 'GPU', 'Ascend'),
                              help='run platform, only support CPU, GPU and Ascend')
    train_parser.add_argument('--dataset_path', type=str, default=r'dataset', help='Dataset Path')
    # 冻结特定层之前的权重，即希望到某一层的权重保持不变
    train_parser.add_argument('--pretrain_ckpt', type=str, default='backbone', choices=['', 'none', 'backbone'],
                              help='freeze the weights of network from start to which layers')
    # 指定参数解释函数为literal_eval，用于将字符串形式的python表达式转换为对应的python对象，以解析布尔值（argparse默认只能解析字符串）
    train_parser.add_argument('--run_distribute', type=ast.literal_eval, default=False,
                              help='Run distribute')
    # 选择是否过滤头部权重，可以减少训练开销，并且防止过拟合
    train_parser.add_argument('--filter_head', type=ast.literal_eval, default=False,
                              help='Filter head weigh parameters when load checkpoint, default is False')
    # 根据以上添加的参数规则解析命令行参数，并将解析的结果存储，以供后续使用
    train_args = train_parser.parse_args()
    train_args.is_training = True
    if train_args.platform == 'CPU':
        train_args.run_distribute = False
    # 返回绝对路径
    train_args.dataset_path = os.path.abspath(train_args.dataset_path)
    return train_args
