
import attr
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from functools import partial

from d_vae.utils import Conv2d


@attr.s(eq=False, repr=False)
class DecoderBlock(nn.Module):
    n_in:     int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 ==0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

    device:        torch.device = attr.ib(default=None)
    requires_grad: bool         = attr.ib(default=False)


    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers ** 2)

        make_conv     = partial(Conv2d, device=self.device, requires_grad=self.requires_grad)
        self.id_path  = make_conv(self.n_in, self.n_out, 1) if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
                ('relu_1', nn.ReLU()),
                ('conv_1', make_conv(self.n_in,  self.n_hid, 1)),
                ('relu_2', nn.ReLU()),
                ('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_3', nn.ReLU()),
                ('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_4', nn.ReLU()),
                ('conv_4', make_conv(self.n_hid, self.n_out, 3)),]))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


"""

"""
@attr.s(eq=False, repr=False)
class Decoder(nn.Module):
    group_count:     int = 4
    n_init:          int = attr.ib(default=128,  validator=lambda i, a, x: x >= 8)
    n_hid:           int = attr.ib(default=256,  validator=lambda i, a, x: x >= 64)
    n_blk_per_group: int = attr.ib(default=2,    validator=lambda i, a, x: x >= 1)
    output_channels: int = attr.ib(default=3,    validator=lambda i, a, x: x >= 1)
    vocab_size:      int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)

    device:              torch.device = attr.ib(default=torch.device('cpu'))
    requires_grad:       bool         = attr.ib(default=False)
    use_mixed_precision: bool         = attr.ib(default=True)


    def __attrs_post_init__(self) -> None:
        super().__init__()

        blk_range  = range(self.n_blk_per_group)
        n_layers   = self.group_count * self.n_blk_per_group
        make_conv  = partial(Conv2d, device=self.device, requires_grad=self.requires_grad)
        make_blk   = partial(DecoderBlock, n_layers=n_layers, device=self.device,
                requires_grad=self.requires_grad)


        """
        blocks.input.w : torch.Size([128, 8192, 1, 1])
        blocks.input.b : torch.Size([128])
        blocks.group_1.block_1.id_path.w : torch.Size([2048, 128, 1, 1])
        blocks.group_1.block_1.id_path.b : torch.Size([2048])
        blocks.group_1.block_1.res_path.conv_1.w : torch.Size([512, 128, 1, 1])
        blocks.group_1.block_1.res_path.conv_1.b : torch.Size([512])
        blocks.group_1.block_1.res_path.conv_2.w : torch.Size([512, 512, 3, 3])
        blocks.group_1.block_1.res_path.conv_2.b : torch.Size([512])
        blocks.group_1.block_1.res_path.conv_3.w : torch.Size([512, 512, 3, 3])
        blocks.group_1.block_1.res_path.conv_3.b : torch.Size([512])
        blocks.group_1.block_1.res_path.conv_4.w : torch.Size([2048, 512, 3, 3])
        blocks.group_1.block_1.res_path.conv_4.b : torch.Size([2048])
        blocks.group_1.block_2.res_path.conv_1.w : torch.Size([512, 2048, 1, 1])
        blocks.group_1.block_2.res_path.conv_1.b : torch.Size([512])
        blocks.group_1.block_2.res_path.conv_2.w : torch.Size([512, 512, 3, 3])
        blocks.group_1.block_2.res_path.conv_2.b : torch.Size([512])
        blocks.group_1.block_2.res_path.conv_3.w : torch.Size([512, 512, 3, 3])
        blocks.group_1.block_2.res_path.conv_3.b : torch.Size([512])
        blocks.group_1.block_2.res_path.conv_4.w : torch.Size([2048, 512, 3, 3])
        blocks.group_1.block_2.res_path.conv_4.b : torch.Size([2048])
        blocks.group_2.block_1.id_path.w : torch.Size([1024, 2048, 1, 1])
        blocks.group_2.block_1.id_path.b : torch.Size([1024])
        blocks.group_2.block_1.res_path.conv_1.w : torch.Size([256, 2048, 1, 1])
        blocks.group_2.block_1.res_path.conv_1.b : torch.Size([256])
        blocks.group_2.block_1.res_path.conv_2.w : torch.Size([256, 256, 3, 3])
        blocks.group_2.block_1.res_path.conv_2.b : torch.Size([256])
        blocks.group_2.block_1.res_path.conv_3.w : torch.Size([256, 256, 3, 3])
        blocks.group_2.block_1.res_path.conv_3.b : torch.Size([256])
        blocks.group_2.block_1.res_path.conv_4.w : torch.Size([1024, 256, 3, 3])
        blocks.group_2.block_1.res_path.conv_4.b : torch.Size([1024])
        blocks.group_2.block_2.res_path.conv_1.w : torch.Size([256, 1024, 1, 1])
        blocks.group_2.block_2.res_path.conv_1.b : torch.Size([256])
        blocks.group_2.block_2.res_path.conv_2.w : torch.Size([256, 256, 3, 3])
        blocks.group_2.block_2.res_path.conv_2.b : torch.Size([256])
        blocks.group_2.block_2.res_path.conv_3.w : torch.Size([256, 256, 3, 3])
        blocks.group_2.block_2.res_path.conv_3.b : torch.Size([256])
        blocks.group_2.block_2.res_path.conv_4.w : torch.Size([1024, 256, 3, 3])
        blocks.group_2.block_2.res_path.conv_4.b : torch.Size([1024])
        blocks.group_3.block_1.id_path.w : torch.Size([512, 1024, 1, 1])
        blocks.group_3.block_1.id_path.b : torch.Size([512])
        blocks.group_3.block_1.res_path.conv_1.w : torch.Size([128, 1024, 1, 1])
        blocks.group_3.block_1.res_path.conv_1.b : torch.Size([128])
        blocks.group_3.block_1.res_path.conv_2.w : torch.Size([128, 128, 3, 3])
        blocks.group_3.block_1.res_path.conv_2.b : torch.Size([128])
        blocks.group_3.block_1.res_path.conv_3.w : torch.Size([128, 128, 3, 3])
        blocks.group_3.block_1.res_path.conv_3.b : torch.Size([128])
        blocks.group_3.block_1.res_path.conv_4.w : torch.Size([512, 128, 3, 3])
        blocks.group_3.block_1.res_path.conv_4.b : torch.Size([512])
        blocks.group_3.block_2.res_path.conv_1.w : torch.Size([128, 512, 1, 1])
        blocks.group_3.block_2.res_path.conv_1.b : torch.Size([128])
        blocks.group_3.block_2.res_path.conv_2.w : torch.Size([128, 128, 3, 3])
        blocks.group_3.block_2.res_path.conv_2.b : torch.Size([128])
        blocks.group_3.block_2.res_path.conv_3.w : torch.Size([128, 128, 3, 3])
        blocks.group_3.block_2.res_path.conv_3.b : torch.Size([128])
        blocks.group_3.block_2.res_path.conv_4.w : torch.Size([512, 128, 3, 3])
        blocks.group_3.block_2.res_path.conv_4.b : torch.Size([512])
        blocks.group_4.block_1.id_path.w : torch.Size([256, 512, 1, 1])
        blocks.group_4.block_1.id_path.b : torch.Size([256])
        blocks.group_4.block_1.res_path.conv_1.w : torch.Size([64, 512, 1, 1])
        blocks.group_4.block_1.res_path.conv_1.b : torch.Size([64])
        blocks.group_4.block_1.res_path.conv_2.w : torch.Size([64, 64, 3, 3])
        blocks.group_4.block_1.res_path.conv_2.b : torch.Size([64])
        blocks.group_4.block_1.res_path.conv_3.w : torch.Size([64, 64, 3, 3])
        blocks.group_4.block_1.res_path.conv_3.b : torch.Size([64])
        blocks.group_4.block_1.res_path.conv_4.w : torch.Size([256, 64, 3, 3])
        blocks.group_4.block_1.res_path.conv_4.b : torch.Size([256])
        blocks.group_4.block_2.res_path.conv_1.w : torch.Size([64, 256, 1, 1])
        blocks.group_4.block_2.res_path.conv_1.b : torch.Size([64])
        blocks.group_4.block_2.res_path.conv_2.w : torch.Size([64, 64, 3, 3])
        blocks.group_4.block_2.res_path.conv_2.b : torch.Size([64])
        blocks.group_4.block_2.res_path.conv_3.w : torch.Size([64, 64, 3, 3])
        blocks.group_4.block_2.res_path.conv_3.b : torch.Size([64])
        blocks.group_4.block_2.res_path.conv_4.w : torch.Size([256, 64, 3, 3])
        blocks.group_4.block_2.res_path.conv_4.b : torch.Size([256])
        blocks.output.conv.w : torch.Size([6, 256, 1, 1])
        blocks.output.conv.b : torch.Size([6])
        """
        self.blocks = nn.Sequential(OrderedDict([
            ('input', make_conv(self.vocab_size, self.n_init, 1)),
            ('group_1', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(self.n_init if i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_2', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(8 * self.n_hid if i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_3', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in blk_range],
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),
            ('group_4', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(2 * self.n_hid if i == 0 else 1 * self.n_hid, 1 * self.n_hid)) for i in blk_range],
            ]))),
            ('output', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
                ('conv', make_conv(1 * self.n_hid, 2 * self.output_channels, 1)),
            ]))),
        ]))



    """
    [8192, 32 32] -> [3, 256, 256]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.vocab_size:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.vocab_size}')
        if x.dtype != torch.float32:
            raise ValueError('input must have dtype torch.float32')

        return self.blocks(x)

