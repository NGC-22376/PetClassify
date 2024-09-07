import os
import torch
from torchinfo import summary
import pandas as pd
import timm
from torchviz import make_dot

# 加载预训练模型
model = timm.create_model("hf_hub:timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k", pretrained=True)

# 创建一个示例输入张量
x = torch.randn(1, 3, 384, 384)

# 获取模型的输出
y = model(x)

# 使用 make_dot 来生成计算图
dot = make_dot(y, params=dict(model.named_parameters()))

# 将计算图保存为文件
dot.render("model_structure", format="png")
# 获取预训练模型储存路径
save_path = os.path.join(os.getcwd(), 'mobilenetv4_hybrid_large_pretrained.pth')

if not os.path.exists(save_path):
    # 获取模型状态字典
    state_dict = model.state_dict()

    # 使用绝对路径保存模型权重
    torch.save(state_dict, save_path)


    # 检查文件是否保存成功
    if os.path.exists(save_path):
        print(f"File saved successfully at: {save_path}")
    else:
        print("File was not saved.")
