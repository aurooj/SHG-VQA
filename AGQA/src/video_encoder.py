import torch
import torch.nn as nn
import timm
from torchvision.models import resnext101_32x8d


class VideoBackbone(nn.Module):

    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone_name = backbone_name

        self.backbone_dict = {
            'slow_r50': self.slow_r50(),

            'slowfast_r50': self.slowfast_r50(),

            'slowfast_r101': self.slowfast_r101(),

            'resnext101': self.resnext101(),

            'video_swin': self.video_swin(),

            'mvit_B': self.mvit_B()

        }

        self.backbone = self.backbone_dict[self.backbone_name]

    def encode(self, x):
        if self.backbone_name == 'resnext101':
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
            x = self.backbone(x)
            x = x.view(b, t, 2048, 7, 7).permute(0, 2, 1, 3, 4)
        else:
            x = self.backbone(x)
        return x

    # def to(self, device='cpu'):
    #     self.backbone = self.backbone.to(device)

    def slow_r50(self):
        if self.backbone_name == 'slow_r50':
            vid_encoder = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
            vid_encoder.blocks[-1] = nn.Identity()  # replacing classification head with identity() block
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Set to eval mode and move to desired device
            # self.vid_encoder = self.vid_encoder.to(device)
            vid_encoder = vid_encoder.eval()
            return vid_encoder

        return False

    def slowfast_r50(self):
        if self.backbone_name == 'slowfast_r50':
            vid_encoder = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
            vid_encoder.blocks[-1] = nn.Identity()  # replacing classification head with identity() block
            vid_encoder.blocks[-2] = nn.Identity() #replacing avg. pooling layer
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Set to eval mode and move to desired device
            # self.vid_encoder = self.vid_encoder.to(device)
            vid_encoder = vid_encoder.eval()
            return vid_encoder

        return False

    def slowfast_r101(self):
        if self.backbone_name == 'slowfast_r101':
            vid_encoder = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=True)
            vid_encoder.blocks[-1] = nn.Identity()  # replacing classification head with identity() block
            vid_encoder.blocks[-2] = nn.Identity()  # replacing avg. pooling layer
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Set to eval mode and move to desired device
            # self.vid_encoder = self.vid_encoder.to(device)
            vid_encoder = vid_encoder.eval()
            return vid_encoder

        return False

    def resnext101(self):
        if self.backbone_name == 'resnext101':
            vid_encoder = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=0)
            vid_encoder.global_pool = nn.Identity()
            # vid_encoder = resnext101_32x8d(pretrained=True)#torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
            # vid_encoder.avgpool = nn.Identity()
            # vid_encoder.fc = nn.Identity()
            vid_encoder = vid_encoder.eval()
            return vid_encoder
        return False

    def video_swin(self):
        if self.backbone_name == 'video_swin':
            raise NotImplementedError
        return False

    def mvit_B(self):
        if self.backbone_name == 'mvit_B':
            #todo: debug for input
            vid_encoder = torch.hub.load('facebookresearch/pytorchvideo', 'mvit_base_32x3', pretrained=True)
            vid_encoder = vid_encoder.eval()
            return vid_encoder
        return False
