"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
import torch
import torch.nn as nn


class EarlyConv(nn.Module):

    def __init__(self, depth=3, in_channels=3, out_channels=None):
        super().__init__()
        self.depth = depth
        channels = [in_channels]
        if out_channels is None:
            out_channels = in_channels
        channels.extend([24*2**i for i in range(depth)])
        self.convs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], 3, 1, 'same'),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU()
                )
                for i in range(depth)
            ]
        )
        self.final = nn.Conv2d(channels[-1], out_channels, 1, 1, 'same')

    def forward(self, x):
        x = self.convs(x)
        x = self.final(x)
        return x


class ModalMixer(nn.Module):

    def __init__(self, modals=['noiseprint', 'bayar', 'srm'], in_channels=[3, 3, 3], out_channels=3):
        super().__init__()

        w = len(modals)
        assert len(modals) == len(in_channels)

        c_h = sum(in_channels)

        self.blocks = nn.ModuleList(
            [
                EarlyConv() for _ in range(w)
            ]
        )
        self.dropout = nn.Dropout(0.33)
        self.mixer = EarlyConv(in_channels=c_h, out_channels=out_channels)

    def forward(self, x):
        m = []
        for m_i, blk in enumerate(self.blocks):
            m.append(blk(x[m_i]))

        x = torch.cat(m, dim=1)
        x = self.dropout(x)
        x = self.mixer(x)
        return x


if __name__ == '__main__':
    x = [torch.zeros((1, 3, 512, 512)) for _ in range(3)]
    model = ModalMixer()
    y = model(x)
    print(y.size())