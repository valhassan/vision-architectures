import segformer
from torchinfo import summary


print("Segformer MIT-B5")
model = segformer.SegFormer("mit_b5", in_channels=3, classes=5)
batch_size = 16
summary(model, input_size=(batch_size, 3, 512, 512))


if __name__ == '__main__':
    pass