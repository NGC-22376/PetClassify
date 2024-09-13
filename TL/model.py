"""
定义网络
"""
import timm
import torch.nn as nn

# 特征抽取模型
model = timm.create_model(
    'mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
    pretrained=True,
    features_only=True,
)


# 自定义MobileNetV4分类器
class MobileNetV4Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV4Classifier, self).__init__()

        # 简单分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，输出1x1的特征图
            nn.Flatten(),  # 展平
            nn.Linear(640, num_classes)  # 最后的特征通道数是640，输出分类数
        )

    def forward(self, x):
        return self.classifier(x)
