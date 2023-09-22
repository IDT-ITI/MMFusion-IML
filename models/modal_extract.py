"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
import torch
import torch.nn as nn
from common.utils import SRMFilter, BayarConv2d
from models.DnCNN_noiseprint import make_net
import logging
import os


class ModalitiesExtractor(nn.Module):
    def __init__(self,
                 modals: list = ('noiseprint', 'bayar', 'srm'),
                 noiseprint_path: str = None):
        super().__init__()
        self.mod_extract = []
        if 'noiseprint' in modals:
            num_levels = 17
            out_channel = 1
            self.noiseprint = make_net(3, kernels=[3, ] * num_levels,
                                  features=[64, ] * (num_levels - 1) + [out_channel],
                                  bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                                  acts=['relu', ] * (num_levels - 1) + ['linear', ],
                                  dilats=[1, ] * num_levels,
                                  bn_momentum=0.1, padding=1)

            if noiseprint_path:
                np_weights = noiseprint_path
                assert os.path.isfile(np_weights)
                dat = torch.load(np_weights, map_location=torch.device('cpu'))
                logging.info(f'Noiseprint++ weights: {np_weights}')
                self.noiseprint.load_state_dict(dat)

            self.noiseprint.eval()
            for param in self.noiseprint.parameters():
                param.requires_grad = False
            self.mod_extract.append(self.noiseprint)
        if 'bayar' in modals:
            self.bayar = BayarConv2d(3, 3, padding=2)
            self.mod_extract.append(self.bayar)
        if 'srm' in modals:
            self.srm = SRMFilter()
            self.mod_extract.append(self.srm)

    def set_train(self):
        if hasattr(self, 'bayar'):
            self.bayar.train()

    def set_val(self):
        if hasattr(self, 'bayar'):
            self.bayar.eval()

    def forward(self, x) -> list:
        out = []
        for mod in self.mod_extract:
            y = mod(x)
            if y.size()[-3] == 1:
                y = torch.tile(y, (3, 1, 1))
            out.append(y)

        return out


if __name__ == '__main__':
    modal_ext = ModalitiesExtractor(['noiseprint', 'bayar', 'srm'], '../pretrained/noiseprint/np++.pth')

    from PIL import Image
    from torchvision.transforms.functional import to_tensor, to_pil_image
    img = Image.open('../data/samples/splicing-01.png').convert('RGB')
    inp = to_tensor(img).unsqueeze(0)

    out = modal_ext(inp)

    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(1, 4)

    ax[0].imshow(img)
    ax[0].set_title('Image')

    noiseprint = out[0][:, 0].squeeze().numpy()
    ax[1].imshow(noiseprint[16:-16:4, 16:-16:4], cmap='gray')
    ax[1].set_title('NoisePrint++')

    bayar = to_tensor(to_pil_image(out[1].squeeze())).permute(1, 2, 0).numpy()
    ax[2].imshow(bayar)
    ax[2].set_title('Bayar')

    srm = to_tensor(to_pil_image(out[2].squeeze())).permute(1, 2, 0).numpy()
    ax[3].imshow(srm)
    ax[3].set_title('SRM')

    plt.show()
