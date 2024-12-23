import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def load_checkpoint(filename):
    ''' Loads checkpoint from provided path
    :param filename: path to checkpoint as .pth.tar or .pth
    :return: (dict) checkpoint ready to be loaded into model instance
    '''
    try:
        print(f"=> loading model '{filename}'\n")
        # For loading external models with different structure in state dict. May cause problems when trying to load optimizer
        checkpoint = torch.load(filename, map_location='cpu')
        # if 'model' not in checkpoint.keys():
        #     temp_checkpoint = {}
        #     temp_checkpoint['model'] = {k: v for k, v in checkpoint.items()}    # Place entire state_dict inside 'model' key
        #     del checkpoint
        #     checkpoint = temp_checkpoint
        return checkpoint
    except FileNotFoundError:
        raise FileNotFoundError(f"=> No model found at '{filename}'")

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoder="mit_b2",
                 in_channels=[64, 128, 320, 512],
                 feature_strides=[4, 8, 16, 32],
                 embedding_dim=768,
                 num_classes=1, dropout_ratio=0.1):
        super(Decoder, self).__init__()
        if encoder == "mit_b0":
            in_channels = [32, 64, 160, 256]
        if encoder in ["mit_b0", "mit_b1"]:
            embedding_dim = 256
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.num_classes = num_classes
        self.in_channels = in_channels
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(dropout_ratio)

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4 = x
        # print(f"c1 shape: {c1.shape}")
        # print(f"c2 shape: {c2.shape}")
        # print(f"c3 shape: {c3.shape}")
        # print(f"c4 shape: {c4.shape}")
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3]).contiguous()
        _c4 = F.interpolate(input=_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        print(f"_c4 shape: {_c4.shape}")

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3]).contiguous()
        _c3 = F.interpolate(input=_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        print(f"_c3 shape: {_c3.shape}")

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3]).contiguous()
        _c2 = F.interpolate(input=_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        print(f"_c2 shape: {_c2.shape}")

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3]).contiguous()
        print(f"_c1 shape: {_c1.shape}")
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        print(f"_c shape: {_c.shape}")

        x = self.dropout(_c)
        x = self.linear_pred(x)
        # print(f"x shape: {x.shape}")

        return x


class SegFormer(nn.Module):
    def __init__(self, encoder, in_channels, classes) -> None:
        super().__init__()
        self.encoder = smp.encoders.get_encoder(name=encoder, in_channels=in_channels, depth=5, drop_path_rate=0.1)
        self.decoder = Decoder(encoder=encoder, num_classes=classes)

    def forward(self, img):
        x = self.encoder(img)[2:]
        x = self.decoder(x)
        x = F.interpolate(input=x, size=img.shape[2:], scale_factor=None, mode='bilinear', align_corners=False)
        return x
    

if __name__ == '__main__':
    # checkpoint_path = "/export/sata01/wspace/test_dir/multi/all_rgb_data/RGB_4class_all_data_b5_VA_20230915.pth.tar"
    # out_path = "/export/sata01/wspace/test_dir/multi/all_rgb_data/encoder_only_weights.pth.tar"
    # mit_b5_encoder_out_path = "/export/sata01/wspace/test_dir/multi/all_rgb_data/mit_b5_encoder.pth.tar"
    # checkpoint = load_checkpoint(checkpoint_path)
    # encoder_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'encoder' in k}
    # torch.save(encoder_state_dict, mit_b5_encoder_out_path)
    # encoder_state_dict = torch.load(mit_b5_encoder_out_path)
    # print(encoder_state_dict.keys())
    # model = SegFormer("mit_b5", in_channels=3, classes=5)
    # model.load_state_dict(encoder_state_dict, strict=False)
    # model_state_dict = model.state_dict()
    # for key, value in encoder_state_dict.items():
    #     assert torch.equal(model_state_dict[key], value), f"Weight mismatch in layer: {key}"
    
    # print("Original weights for patch_embed1:", encoder_state_dict['encoder.patch_embed1.proj.weight'][0][0][1])
    # print("Loaded weights for patch_embed1:",  model_state_dict['encoder.patch_embed1.proj.weight'][0][0][1])
    
    # torch.save(model.state_dict(), out_path)
    # checkpoint = torch.load(out_path)
    # print(checkpoint.keys())
    
    model = SegFormer("mit_b5", in_channels=3, classes=5)
    batch_size = 8
    x = torch.rand(batch_size, 3, 512, 512)
    out = model(x)