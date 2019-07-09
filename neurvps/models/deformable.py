import os
import math
import warnings
from glob import glob

import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable


def load_cpp_ext(ext_name):
    root_dir = os.path.join(os.path.split(__file__)[0])
    src_dir = os.path.join(root_dir, "cpp")
    tar_dir = os.path.join(src_dir, "build", ext_name)
    os.makedirs(tar_dir, exist_ok=True)
    srcs = glob(f"{src_dir}/*.cu") + glob(f"{src_dir}/*.cpp")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torch.utils.cpp_extension import load

        ext = load(
            name=ext_name,
            sources=srcs,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[],
            build_directory=tar_dir,
        )
    return ext


# defer calling load_cpp_ext to make CUDA_VISIBLE_DEVICES happy
DCN = None


class DeformConvFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        weight,
        bias,
        stride,
        padding,
        dilation,
        group,
        deformable_groups,
        im2col_step,
    ):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        output = DCN.deform_conv_forward(
            input,
            weight,
            bias,
            offset,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.group,
            ctx.deformable_groups,
            ctx.im2col_step,
        )
        ctx.save_for_backward(input, offset, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_weight, grad_bias = DCN.deform_conv_backward(
            input,
            weight,
            bias,
            offset,
            grad_output,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.group,
            ctx.deformable_groups,
            ctx.im2col_step,
        )

        return (
            grad_input,
            grad_offset,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        deformable_groups=1,
        im2col_step=11,
        bias=True,
    ):
        global DCN
        DCN = load_cpp_ext("DCN")
        super(DeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError(
                "in_channels {} must be divisible by groups {}".format(
                    in_channels, groups
                )
            )
        if out_channels % groups != 0:
            raise ValueError(
                "out_channels {} must be divisible by groups {}".format(
                    out_channels, groups
                )
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            if self.use_bias:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                nn.init.zeros_(self.bias)

    def forward(self, input, offset):
        assert (
            2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            == offset.shape[1]
        )
        return DeformConvFunction.apply(
            input.contiguous(),
            offset.contiguous(),
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
            self.im2col_step,
        )
