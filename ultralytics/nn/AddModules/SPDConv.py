import torch
import torch.nn as nn
import torch.nn.functional as F
 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
class SPDConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        c1 = c1 * 4
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def _space_to_depth(self, x):
        """四向子采样拼通道；奇数 H/W 时先右下 pad 1，避免四个张量空间尺寸不一致。"""
        _, _, h, w = x.shape
        if h % 2 == 1 or w % 2 == 1:
            x = F.pad(x, (0, w % 2, 0, h % 2))  # (left,right,top,bottom)
        return torch.cat(
            [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1
        )

    def forward(self, x):
        """Space-to-depth 下采样后卷积 + BN + 激活。"""
        x = self._space_to_depth(x)
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        x = self._space_to_depth(x)
        return self.act(self.conv(x))
 
 
    