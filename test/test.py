from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'mobilenetv4_hybrid_large.e600_r384_in1k',
    pretrained=True,
    features_only=True,
)
model = model.eval()


# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 24, 192, 192])
    #  torch.Size([1, 48, 96, 96])
    #  torch.Size([1, 96, 48, 48])
    #  torch.Size([1, 192, 24, 24])
    #  torch.Size([1, 960, 12, 12])

    print(o.shape)
