import torch.nn as nn


def make_divisible(
        value: float,
        divisor: int,
        min_value=None,
        round_down_protect: bool = True,
) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)

    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    # 创建一个 nn.Sequential 容器，用于按顺序添加多个网络层
    conv = nn.Sequential()

    # 计算卷积层的填充大小，确保输入和输出的宽高相同（在步幅为 1 的情况下）
    padding = (kernel_size - 1) // 2

    # 添加二维卷积层
    # inp: 输入通道数，oup: 输出通道数，kernel_size: 卷积核大小
    # stride: 步幅，padding: 填充，bias: 是否添加偏置，groups: 组卷积的组数
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))

    # 如果 norm 为 True，添加批归一化层
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))

    # 如果 act 为 True，添加 ReLU6 激活函数
    if act:
        conv.add_module('Activation', nn.ReLU6())

    # 返回 nn.Sequential 容器，包含卷积层、可选的批归一化层和激活函数
    return conv


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 inp,  # 输入通道数
                 oup,  # 输出通道数
                 start_dw_kernel_size,  # 起始深度卷积的核大小
                 middle_dw_kernel_size,  # 中间深度卷积的核大小
                 middle_dw_downsample,  # 是否在中间深度卷积处进行下采样
                 stride,  # 卷积的步幅
                 expand_ratio  # 通道扩展的比例
                 ):

        super().__init__()  # 初始化 nn.Module 的父类

        # 起始的深度卷积 
        self.start_dw_kernel_size = start_dw_kernel_size  # 保存起始深度卷积核大小 
        if self.start_dw_kernel_size:  # 如果指定了起始深度卷积 
            stride_ = stride if not middle_dw_downsample else 1  # 如果中间深度卷积不下采样，保持 stride，否则将 stride 设为 1 
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
            # 创建起始的深度卷积层，输入输出通道数相同，groups=inp表示是深度卷积 

        # 使用 1x1 卷积进行通道扩展 
        expand_filters = make_divisible(inp * expand_ratio, 8)  # 计算扩展后的通道数，并使其可被 8 整除 
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # 创建 1x1 卷积层用于通道扩展 

        # 中间的深度卷积 
        self.middle_dw_kernel_size = middle_dw_kernel_size  # 保存中间深度卷积核大小
        if self.middle_dw_kernel_size:  # 如果指定了中间深度卷积 
            stride_ = stride if middle_dw_downsample else 1  # 决定是否在中间深度卷积时应用 stride 
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
            # 创建中间的深度卷积层，使用扩展后的通道数，groups=expand_filters表示是深度卷积 

        # 使用 1x1 卷积进行通道压缩 (Projection with 1x1 convolution)
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)
        # 创建 1x1 卷积层用于通道压缩，将通道数投影回所需的输出通道数

    def forward(self, x):
        if self.start_dw_kernel_size:  # 如果有起始深度卷积 
            x = self._start_dw_(x)  # 应用起始深度卷积
            # print("_start_dw_", x.shape) 

        x = self._expand_conv(x)  # 应用通道扩展的 1x1 卷积
        # print("_expand_conv", x.shape)

        if self.middle_dw_kernel_size:  # 如果有中间深度卷积
            x = self._middle_dw(x)  # 应用中间深度卷积 
            # print("_middle_dw", x.shape)  

        x = self._proj_conv(x)  # 应用通道压缩的 1x1 卷积 
        # print("_proj_conv", x.shape)  

        return x  # 返回处理后的输出


class FusedIB(nn.Module):
    def __init__(self, input_dim, output_dim, expand_ratio):
        super().__init__()
        expand_dim = make_divisible(input_dim * (expand_ratio + 1), 8)
        # 通道拓展和深度卷积二合一
        self.conv = conv_2d(input_dim, expand_dim, stride=2)
        # 逐点卷积将通道数映射回要求输出
        self.proj_conv = conv_2d(expand_dim, output_dim, kernel_size=1, act=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.proj_conv(x)
        return x
