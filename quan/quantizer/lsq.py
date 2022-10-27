import torch as t

from .quantizer import Quantizer


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)
        self.bit=bit
        #print('bit in lsq ',bit)
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones(1)*20/ self.thd_pos)
        self.inited = False
        self.collect_stats = False
        self.max=0


    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            self.s = t.nn.Parameter(x.detach().abs().max() *2/ (self.thd_pos))
        self.inited=True

    def forward(self, x):
        if self.collect_stats:
            self.max = max(self.max,x.detach().abs().max())
            self.s = t.nn.Parameter(self.max * 2 / (self.thd_pos))

        if not self.inited:#todo: (try mean?), do all DS for activisions ONLY!!, try per channel
            self.s = t.nn.Parameter(x.detach().abs().max() *2/ (self.thd_pos))
            self.inited=True
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)
        # print(f'BEFORE: std={x.std().item()}, min={x.min().item()}, max={x.max().item()} ')
        x = x / s_scale

        # print( x_i.abs().max().item(),1/s_scale,x.abs().max().item(), self.thd_pos)
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        # print('cl',((x_i-x*s_scale)**2).mean()**(1/2))
        x = round_pass(x)
        x = x * s_scale
        # print('rnd',((x_i-x*s_scale)**2).mean()**(1/2))
        # print(f'AFTER: std={x.std().item()}, min={x.min().item()}, max={x.max().item()} ')
        return x