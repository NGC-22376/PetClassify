"""
定义网络
"""
import timm
import torch.nn as nn
from src.UIB import UniversalInvertedBottleneckBlock as UIB
from src.UIB import conv_2d as conv
from src.UIB import FusedIB

# 特征抽取器
model = timm.create_model(
    # 'mobilenetv4_conv_small.e2400_r224_in1k',
    'mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
    pretrained=True,
    features_only=True,
)


def print_layer_shape(module, input, output):
    print(f"{module.__class__.__name__} - Input shape: {input[0].shape}, Output shape: {output.shape}")


# 自定义MobileNetV4分类器
class MobileNetV4Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 简单分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，输出1x1的特征图
            nn.Flatten(),  # 展平
            nn.Linear(640, num_classes)  # 最后的特征通道数是640，输出分类数
        )

        # MNV4_S
        self.mnv4_S = nn.Sequential(
            conv(3, 32, 3, 2),
            self.fused_IB(32, 32, 1),
            self.fused_IB(32, 64, 3),
            self.extra_depthwise(64, 96, 5, 5, 1, 2, 3),
            nn.Dropout(0.5),
            self.inverted_bottleneck(96, 96, 3, 1, 2),
            self.inverted_bottleneck(96, 96, 3, 1, 2),
            nn.Dropout(0.5),
            self.inverted_bottleneck(96, 96, 3, 1, 2),
            self.inverted_bottleneck(96, 96, 3, 1, 2),
            self.conv_next(96, 96, 3, 1, 4),
            nn.Dropout(0.5),
            self.extra_depthwise(96, 128, 3, 3, 1, 2, 6),
            self.extra_depthwise(128, 128, 5, 5, 1, 1, 4),
            self.inverted_bottleneck(128, 128, 5, 1, 4),
            self.inverted_bottleneck(128, 128, 5, 1, 3),
            nn.Dropout(0.5),
            self.inverted_bottleneck(128, 128, 3, 1, 4),
            self.inverted_bottleneck(128, 128, 3, 1, 4),
            nn.Dropout(0.5),
            conv(128, 960, 1, 1),
            nn.AvgPool2d(7),
            conv(960, 1280, 1, 1),
            nn.Dropout(0.5),
            conv(1280, num_classes, 1, 1)
        )
        # # 对 Sequential 中的每一层注册钩子
        # for layer in self.mnv4_S:
        #     layer.register_forward_hook(print_layer_shape)

    def extra_depthwise(self, in_dim, out_dim, start_kernel, middle_kernel, middle_down_sample, stride, expand_ratio):
        return UIB(in_dim, out_dim, start_kernel, middle_kernel, middle_down_sample, stride, expand_ratio)

    def inverted_bottleneck(self, in_dim, out_dim, middle_kernel, stride, expand_ratio):
        return UIB(in_dim, out_dim, 0, middle_kernel, 1, stride, expand_ratio)

    def conv_next(self, in_dim, out_dim, start_kernel, stride, expand_ratio):
        return UIB(in_dim, out_dim, start_kernel, 0, 0, stride, expand_ratio)

    def FFN(self, in_dim, out_dim, stride, expand_ratio):
        return UIB(in_dim, out_dim, 0, 0, 0, stride, expand_ratio)

    def fused_IB(self, in_dim, out_dim, expand_ratio):
        return FusedIB(in_dim, out_dim, expand_ratio)

    def forward(self, x):
        # # 特征
        # return self.classifier(x)
        x = self.mnv4_S(x)
        x = x.view(x.size(0), -1)
        return x
