"""
定义网络
"""
import timm
import torch
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
            nn.Linear(960, num_classes)  # 最后的特征通道数是960，输出分类数
        )

    def forward(self, x):
        return self.classifier(x)


def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {file_path}, starting from epoch {epoch}")
    return epoch
