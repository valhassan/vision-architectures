import torch
import segformer
from torchinfo import summary


print("Segformer MIT-B5")
model = segformer.SegFormer("mit_b5", in_channels=3, classes=5)
batch_size = 8
x = torch.rand(batch_size, 3, 512, 512)
out = model(x)
# summary(model, input_data=[x])


if __name__ == '__main__':
    pass