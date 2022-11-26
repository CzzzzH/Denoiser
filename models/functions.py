import torch 
from . import halide_ops as ops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Scatter2Gather(torch.autograd.Function):
    """Converts (transposes) scatter kernels into gather kernels.

    Kernel weights at (x, y) for offset (dx, dy) (i.e. scatter[., dy, dx, y,
    x]) are put at gather[., -dy, -dx, y+dy, x+dx].

    Args:
      data(th.Tensor)[bs, k_h, k_w, h, w]: scatter kernel weights.

    Returns:
      (th.Tensor)[bs, k_h, k_w, h, w]: gather kernel weights.
    """
    @staticmethod
    def forward(ctx, data):
        output = data.new()
        output.resize_as_(data)
        
        if device == 'cpu': ops.scatter2gather_cpu(data, output)
        else: ops.scatter2gather_cuda(data, output)
            
        return output

    @staticmethod
    def backward(ctx, d_output):
        d_data = d_output.new()
        d_data.resize_as_(d_output)
        
        if device == 'cpu': ops.scatter2gather_cpu(d_output, d_data)
        else: ops.scatter2gather_cuda(d_output, d_data)
        
        return d_data
    

class KernelWeighting(torch.autograd.Function):
    """Locally-weighted average of the input values using kernel weights.

    Args:
      data(th.Tensor)[bs, c, h, w]: input values to be locally averaged.
          weights(th.Tensor)[bs, k_h, k_w, h, w]: kernel weights. k_h, k_w are
          the kernel's dimensions. Channels are filtered independently.

    Returns:
      output(th.Tensor)[bs, c, h, w]: weighted average of data using weights.
          output[., c, y, x] = sum_{dx, dy} weights[., dy, dx, x, y]*data[., c,
          y+dy, x+dx].
      sum_w(th.Tensor)[bs, h, w]: sum of weights per pixel
    """
    @staticmethod
    def forward(ctx, data, weights):
        
        bs, c, h, w = data.shape
        
        output = data.new()
        sum_w = data.new()
        output.resize_as_(data)
        sum_w.resize_(bs, h, w)
        
        if device == 'cpu': ops.kernel_weighting_cpu(data, weights, output, sum_w)
        else: ops.kernel_weighting_cuda(data, weights, output, sum_w)

        ctx.save_for_backward(data, weights, sum_w)
        return output, sum_w

    @staticmethod
    def backward(ctx, d_output, d_sum_w):
        
        data, weights, sum_w = ctx.saved_tensors
        
        d_data = data.new()
        d_weights = weights.new()
        d_data.resize_as_(data)
        d_weights.resize_as_(weights)
        
        if device == 'cpu': ops.kernel_weighting_grad_cpu(data, weights, sum_w, d_output, d_sum_w, d_data, d_weights)
        else: ops.kernel_weighting_grad_cuda(data, weights, sum_w, d_output, d_sum_w, d_data, d_weights)

        return d_data, d_weights
