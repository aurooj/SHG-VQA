import torch
from torchvision import transforms
from torchvision.transforms import Compose, Lambda
from src.param import args
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
    Normalize,
    AugMix,
    RandAugment,
    Permute
)

CLIP_LEN = args.CLIP_LEN
mean = {
    'slow_r50': [0.45, 0.45, 0.45],
    'slowfast_r50': [0.45, 0.45, 0.45],
    'slowfast_r101':[0.45, 0.45, 0.45],
    'resnext101': [0.485, 0.456, 0.406],
    'video_swin': [],
    'mvit_B': [0.45, 0.45, 0.45]
}

std = {
    'slow_r50': [0.225, 0.225, 0.225],
    'slowfast_r50': [0.225, 0.225, 0.225],
    'slowfast_r101':[0.225, 0.225, 0.225],
    'resnext101': [0.229, 0.224, 0.225],
    'video_swin': [],
    'mvit_B': [0.225, 0.225, 0.225]
}

color_jitter = (0, 0, 0)
crop_size = 256
sampling_rate = 2
alpha = 4


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list



class DataTransforms:

    def __init__(self, transform_opt: str):
        self.transform_opt = transform_opt
        self.do_augment = self.transform_opt in ['aug_mix', 'rand_aug']

        self.transform_dict = {
            'no_aug': Compose(
                [
                    Permute((3, 0, 1, 2)),
                    UniformTemporalSubsample(CLIP_LEN),
                    transforms.Resize(size=(224, 224)),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean[args.backbone], std[args.backbone]),
                ]
            ),
            'no_aug_slowfast': Compose(
                [
                    Permute((3, 0, 1, 2)),
                    UniformTemporalSubsample(CLIP_LEN),
                    transforms.Resize(size=(256, 256)),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean[args.backbone], std[args.backbone]),
                    PackPathway()
                ]
            ),

            'aug_mix': Compose(
                [
                    Permute((3, 0, 1, 2)),
                    UniformTemporalSubsample(CLIP_LEN),
                    transforms.Resize(size=(224, 224)),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean[args.backbone], std[args.backbone]),
                    Permute((1, 0, 2, 3)),
                    AugMix(),
                    Permute((1, 0, 2, 3))
                ]
            ),

            'rand_aug': Compose(
                [
                    Permute((3, 0, 1, 2)),
                    UniformTemporalSubsample(CLIP_LEN),
                    transforms.Resize(size=(224, 224)),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean[args.backbone], std[args.backbone]),
                    Permute((1, 0, 2, 3)),
                    RandAugment(),
                    Permute((1, 0, 2, 3))
                ]
            ),
            'rand_aug_slowfast': Compose(
                [
                    Permute((3, 0, 1, 2)),
                    UniformTemporalSubsample(CLIP_LEN),
                    transforms.Resize(size=(256, 256)),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean[args.backbone], std[args.backbone]),
                    Permute((1, 0, 2, 3)),
                    RandAugment(),
                    Permute((1, 0, 2, 3)),
                    PackPathway()

                ]
            )
        }

        self.transform_fn = self.transform_dict[self.transform_opt]

    def transform(self, x):
        return self.transform_fn(x)


class QAInputArrange:

    def __init__(self, qa_arrange_opt: str):
        self.qa_arrange_opt = qa_arrange_opt

        assert qa_arrange_opt in ['add_sep_all', 'no_sep_all', 'add_sep', 'no_sep']

        space = ' '

        self.qa_process_dict = {
            'add_sep_all': lambda x, y: x + ' [SEP] ' + ' '.join([f' {k}: {v} [SEP]' for k, v in y.items()]),
            'no_sep_all': lambda x, y: x + space + space.join([f' {k}: {v}' for k, v in y.items()]),
            'no_sep': lambda x, y: x + space + y,
            'add_sep': lambda x, y: x + ' [SEP] ' + y
        }

    def qa_prep(self, q, choices):

        if self.qa_arrange_opt in ['add_sep_all', 'no_sep_all']:
            out = self.qa_process_dict[self.qa_arrange_opt](q, choices)
            return out

        elif self.qa_arrange_opt in ['add_sep', 'no_sep']:
            out = {}
            for k, v in choices.items():
                ch = f'{k}: {v}'
                out[f'qa{k}'] = self.qa_process_dict[self.qa_arrange_opt](q, ch)

            return out
