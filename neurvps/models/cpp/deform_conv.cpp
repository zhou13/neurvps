#include "deform_conv_cpu.h"
#include "deform_conv_cuda.h"

at::Tensor
deform_conv_forward(const at::Tensor &input,
               const at::Tensor &weight,
               const at::Tensor &bias,
               const at::Tensor &offset,
               const int kernel_h,
               const int kernel_w,
               const int stride_h,
               const int stride_w,
               const int pad_h,
               const int pad_w,
               const int dilation_h,
               const int dilation_w,
               const int group,
               const int deformable_group,
               const int im2col_step)
{
    if (input.type().is_cuda())
    {
        return deform_conv_cuda_forward(input, weight, bias, offset,
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w,
                                   group,
                                   deformable_group,
                                   im2col_step);
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
deform_conv_backward(const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &bias,
                const at::Tensor &offset,
                const at::Tensor &grad_output,
                const int kernel_h,
                const int kernel_w,
                const int stride_h,
                const int stride_w,
                const int pad_h,
                const int pad_w,
                const int dilation_h,
                const int dilation_w,
                const int group,
                const int deformable_group,
                const int im2col_step)
{
    if (input.type().is_cuda())
    {
        return deform_conv_cuda_backward(input,
                                    weight,
                                    bias,
                                    offset,
                                    grad_output,
                                    kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    group,
                                    deformable_group,
                                    im2col_step);
    }
    AT_ERROR("Not implemented on the CPU");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deform_conv_forward", &deform_conv_forward, "Backward pass of deformable convolution");
    m.def("deform_conv_backward", &deform_conv_backward, "Backward pass of deformable convolution");
}
