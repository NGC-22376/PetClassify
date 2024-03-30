import os
import random
import time

import numpy as np
from mindspore import Tensor
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.train.callback import Callback
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.serialization import save_checkpoint

from .mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2


class CrossEntropyWithLabelSmooth(_Loss):
    """
    CrossEntropyWith LabelSmooth.交叉熵损失函数 + 标签平滑——抑制过拟合发生

    Args:
        smooth_factor (float): smooth factor. Default is 0.
        num_classes (int): number of classes. Default is 1000.

    Returns:
        None.

    Examples:
        >>> CrossEntropyWithLabelSmooth(smooth_factor=0., num_classes=1000)
    """

    def __init__(self, smooth_factor=0., num_classes=1000):
        super(CrossEntropyWithLabelSmooth, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor /
                                (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean(False)
        self.cast = P.Cast()

    def construct(self, logit, label):
        one_hot_label = self.onehot(self.cast(label, mstype.int32), F.shape(logit)[1],
                                    self.on_value, self.off_value)
        out_loss = self.ce(logit, one_hot_label)
        out_loss = self.mean(out_loss, 0)
        return out_loss


class Monitor(Callback):
    """
    Monitor loss and time.
    监视模型训练过程中的指标，损失函数的变化、训练时间，根据这些指标进行一些操作，如学习率调整、模型保存等。

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> Monitor(100,lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}".format(epoch_mseconds,
                                                                                      per_step_mseconds,
                                                                                      np.mean(self.losses)))

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.3f}/{:5.3f}], time:[{:5.3f}], lr:[{:5.3f}]".format(
            cb_params.cur_epoch_num -
            1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]))


def acc(feature, label, model):
    """计算模型正确率的函数"""
    correct_num = 0
    total_num = 0
    for j in range(len(feature)):
        x = Tensor(feature[j])
        y = label[j]
        y_pred = model.predict(x).asnumpy()
        indices = y_pred.argmax(axis=1)
        result = (np.equal(indices, y) * 1).reshape(-1)
        correct_num += result.sum()
        total_num += result.shape[0]
    return correct_num / total_num


def load_ckpt(network, pretrain_ckpt_path, trainable=True):
    """load checkpoint into network.通常与迁移学习有关"""
    param_dict = load_checkpoint(pretrain_ckpt_path)
    load_param_into_net(network, param_dict)
    if not trainable:
        for param in network.get_parameters():
            param.requires_grad = False


def define_net(config, activation="None"):
    backbone_net = MobileNetV2Backbone()
    head_net = MobileNetV2Head(input_channel=backbone_net.out_channels,
                               num_classes=config.num_classes)
    net = mobilenet_v2(backbone_net, head_net, activation)
    return backbone_net, head_net, net


def get_networks(net, loss, opt):
    # 创建带有损失函数的网络
    network = WithLossCell(net, loss)
    # 创建用于单步训练的网络，和优化器opt捆绑，方便在每个训练步骤中方便地执行前向传播、反向传播和参数更新
    network = TrainOneStepCell(network, opt)
    # 将网络设置为训练模式
    network.set_train()
    # 创建包含自定义神经网络的模型
    model = Model(net)
    return network, model


def train(network, model, net, data, config):
    best_acc = {"epoch": 0, "acc": 0}
    save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_0')
    if not os.path.isdir(save_ckpt_path):
        os.mkdir(save_ckpt_path)
    for epoch in range(config.epoch_size):
        train_features, train_labels, eval_features, eval_labels = data
        # 是否提前停止训练
        early_stop = False
        # 训练的次数
        step_size = len(train_features)
        # 生成索引列表并打乱
        idx_list = list(range(step_size))
        random.shuffle(idx_list)
        epoch_start = time.time()
        losses = []
        for j in idx_list:
            feature = Tensor(train_features[j])
            label = Tensor(train_labels[j])
            # 前向传播 + 损失值计算
            losses.append(network(feature, label).asnumpy())
        train_acc = acc(train_features, train_labels, model)
        eval_acc = acc(eval_features, eval_labels, model)
        epoch_mseconds = (time.time() - epoch_start) * 1000
        per_step_mseconds = epoch_mseconds / step_size
        print(
            "epoch[{}/{}], iter[{}] cost: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}, acc: {:5.3f}, "
            "train acc: {:5.3f}".format(epoch + 1, config.epoch_size, step_size, epoch_mseconds, per_step_mseconds,
                                        np.mean(np.array(losses)),
                                        eval_acc, train_acc))
        acc_f = eval_acc

        if acc_f >= best_acc["acc"]:
            print("update best acc:", acc_f)
            # 保存最佳模型
            best_acc_path = os.path.join(save_ckpt_path, "mobilenetv2_best.ckpt")
            save_checkpoint(net, best_acc_path)
            best_acc = {"epoch": epoch, "acc": acc_f}
        # 当连续多个模型没有性能提升时，中断训练并恢复到最佳模型
        if epoch - best_acc["epoch"] > 10:
            print(f"early stop! the best epoch is {best_acc['epoch'] + 1}")
            # load the parameters with best acc into net from pre_training checkpoint
            load_ckpt(net, os.path.join(save_ckpt_path, "mobilenetv2_best.ckpt"))
            break
