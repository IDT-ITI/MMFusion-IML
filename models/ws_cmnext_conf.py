"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.base import BaseModel
from models.heads import SegFormerHead
from models.layers import trunc_normal_
import logging
from models.backbones import *


class WSCMNeXtWithConf(BaseModel):
    def __init__(self, cfg=None) -> None:
        backbone = cfg.BACKBONE
        num_classes = cfg.NUM_CLASSES
        modals = cfg.MODALS
        logging.info('Currently training for {}'.format(cfg.TRAIN_PHASE))
        logging.info('Loading Model: {}, with backbone: {}'.format(cfg.NAME, cfg.BACKBONE))
        super().__init__(backbone, num_classes, modals)
        extra_modals = modals[1:]
        num_extra_modals = len(extra_modals)
        channels = [c * num_extra_modals for c in self.backbone.channels]

        self.dropout = nn.ModuleList([nn.Dropout(0.33) for _ in range(4)])

        self.decode_head = SegFormerHead(channels, 256 if 'B0' in backbone or 'B1' in backbone else 512,
                                         num_classes)
        self.conf_head = SegFormerHead(channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, 1)

        if cfg.DETECTION == 'confpool':
            self.detection = nn.Sequential(
                            nn.Linear(in_features=8, out_features=128),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(in_features=128, out_features=1),
                            )
        self.apply(self._init_weights)
        self.train_phase = cfg.TRAIN_PHASE
        assert self.train_phase in ['localization', 'detection']
        self.init_pretrained(cfg.PRETRAINED, backbone)
        if self.train_phase == 'detection':
            self.backbone.eval()
            self.dropout.eval()
            self.decode_head.eval()
            self.conf_head.train()
            self.detection.train()
            for p in self.decode_head.parameters():
                p.requires_grad = False
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.eps = 0.001
            m.momentum = 0.1
    def set_train(self):
        if self.train_phase == 'localization':
            self.backbone.train()
            self.dropout.train()
            self.decode_head.train()
        elif self.train_phase == 'detection':
            self.conf_head.train()
            self.detection.train()
        else:
            raise ValueError(f'Train phase {self.train_phase} not recognized!')

    def set_val(self):
        if self.train_phase == 'localization':
            self.backbone.eval()
            self.dropout.eval()
            self.decode_head.eval()
        elif self.train_phase == 'detection':
            self.conf_head.eval()
            self.detection.eval()
        else:
            raise ValueError(f'Train phase {self.train_phase} not recognized!')

    def forward(self, x: list):
        y = self.backbone(x)
        # y = self.dropout(y)
        for i, _ in enumerate(y):
            y[i] = self.dropout[i](y[i])
        out = self.decode_head(y)
        out = F.interpolate(out, size=x[0].shape[2:], mode='bilinear', align_corners=False)
        if self.train_phase == 'detection':
            # for i, _ in enumerate(y):
            #     y[i] = self.dropout_conf[i](y[i])
            conf = self.conf_head(y)
            conf = F.interpolate(conf, size=x[0].shape[2:], mode='bilinear', align_corners=False)
            from .layer_utils import weighted_statistics_pooling
            f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
            f2 = weighted_statistics_pooling(out[:, 1:2, :, :] - out[:, 0:1, :, :], F.logsigmoid(conf)).view(
                out.shape[0], -1)
            det = self.detection(torch.cat((f1, f2), -1))
            return out, conf, det

        return out

    def init_pretrained(self, pretrained: str = None, backbone: str = None) -> None:
        if pretrained:
            logging.info('Loading pretrained module: {}'.format(pretrained))

            load_dualpath_model(self.backbone, pretrained, backbone)


def load_dualpath_model(model, model_file, backbone):
    extra_pretrained = model_file if 'MHSA' in backbone else None
    if isinstance(extra_pretrained, str):
        raw_state_dict_ext = torch.load(extra_pretrained, map_location=torch.device('cpu'))
        if 'state_dict' in raw_state_dict_ext.keys():
            raw_state_dict_ext = raw_state_dict_ext['state_dict']
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
            for i, _ in enumerate(model.modals):
                ex = k.replace('patch_embed', 'extra.' + str(i) + '.patch_embed')
                state_dict[ex] = v

        elif k.find('block') >= 0:
            state_dict[k] = v
            for i, _ in enumerate(model.modals):
                ex = k.replace('block', 'extra.' + str(i) + '.block')
                state_dict[ex] = v

        elif k.find('norm') >= 0:
            state_dict[k] = v
            for i, _ in enumerate(model.modals):
                ex = k.replace('norm', 'extra.' + str(i) + '.norm')
                state_dict[ex] = v


    msg = model.load_state_dict(state_dict, strict=False)
    del state_dict


if __name__ == '__main__':
    from configs.cmnext_init_cfg import _C as cfg
    logging.basicConfig(level=getattr(logging, 'INFO'))
    cfg.MODEL.PRETRAINED = '../pretrained/segformer/mit_b2.pth'
    cfg.MODEL.BACKBONE = 'WSCMNeXtMHSA-B2'
    model = WSCMNeXtWithConf(cfg.MODEL)
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024) * 2,
         torch.ones(1, 3, 1024, 1024) * 3]
    y = model(x)
    print(y.shape)