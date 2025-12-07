import torch
import torch.nn as nn
from quantart.components.blocks import ResnetBlock, AttnBlock


class StyleTransferBlock(nn.Module):
    def __init__(self, channels, residual=True, use_conv=True, use_selfatt=True):
        super().__init__()
        self.use_conv = use_conv
        self.use_selfatt = use_selfatt
        if use_conv:
            self.res_block = ResnetBlock(in_channels=channels,
                                         out_channels=channels,
                                         temb_channels=0,
                                         dropout=0.0)
        if use_selfatt:
            self.self_attn_block = AttnBlock(channels, residual=residual)
        self.attn_block = AttnBlock(channels, residual=residual)

    def forward(self, x, ref):
        if self.use_conv:
            x = self.res_block(x)
        if self.use_selfatt:
            x = self.self_attn_block(x)
        if ref is not None:
            x = self.attn_block(x, ref)
        return x


class StyleTransferModule(nn.Module):
    def __init__(self, channels, block_num=6, residual=True, use_conv=True, use_selfatt=True):
        super().__init__()
        blocks = []
        for i in range(block_num):
            blocks.append(StyleTransferBlock(
                channels, residual=residual, use_conv=use_conv, use_selfatt=use_selfatt))
        self.blocks = torch.nn.Sequential(*blocks)

    def forward(self, x, ref, return_all=False):
        x_list = []
        for block in self.blocks:
            x = block(x, ref)
            x_list.append(x)
        if return_all:
            return x_list
        return x
