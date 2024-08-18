import timm
import torch
import os

# 加载预训练模型
model = timm.create_model("hf_hub:timm/mobilenetv4_hybrid_large.ix_e600_r384_in1k", pretrained=True)

# 获取模型状态字典
state_dict = model.state_dict()

# 使用绝对路径保存模型权重
save_path = os.path.join(os.getcwd(), 'mobilenetv4_hybrid_large_pretrained.pth')
torch.save(state_dict, save_path)

# 检查文件是否保存成功
if os.path.exists(save_path):
    print(f"File saved successfully at: {save_path}")
else:
    print("File was not saved.")
