import torch
import torch.nn.functional as F

from skimage.metrics import structural_similarity as ssim

loss_eps = 1e-12

def L1_loss(pred, gt):
    return F.l1_loss(pred, gt)

def L2_loss(pred, gt):
    return F.mse_loss(pred, gt)

def relative_L1_loss(pred, gt):
    mae = F.l1_loss(pred, gt)
    return mae / (torch.abs(pred).sum() + loss_eps)

def relative_L2_loss(pred, gt):
    mse = F.mse_loss(pred, gt)
    return mse / ((pred ** 2).sum() + loss_eps)

def RMSE(pred, gt):
    return torch.sqrt(F.mse_loss(pred, gt))

def SSIM(pred, gt, flag=False):
    
    if flag:
        return 1 - ssim(pred, gt, channel_axis=2)
    else:
        channel = pred.shape[0] * pred.shape[1]
        pred = pred.reshape(channel, pred.shape[2], pred.shape[3])
        gt = gt.reshape(channel, gt.shape[2], gt.shape[3])
        return 1 - ssim(pred, gt, channel_axis=0)

def _tonemap(im):
    """Helper Reinhards tonemapper.

    Args:
        im(th.Tensor): image to tonemap.

    Returns:
        (th.Tensor) tonemaped image.
    """
    im = torch.clamp(im, min=0)
    return im / (1 + im)

class TonemappedRelativeMSE(torch.nn.Module):
    """Relative mean-squared error on tonemaped images.

    Args:
        eps(float): small number to avoid division by 0.
    """
    def __init__(self, eps=1e-2):
        super(TonemappedRelativeMSE, self).__init__()
        self.eps = eps  # avoid division by zero

    def forward(self, im, ref):
        im = _tonemap(im)
        ref = _tonemap(ref)
        mse = torch.pow(im - ref, 2)
        loss = mse / (torch.pow(ref, 2) + self.eps)
        loss = 0.5 * torch.mean(loss)
        return loss


if __name__ == "__main__":
    
    pred = torch.Tensor([[0., 1.], [1., 0.]])
    gt = torch.Tensor([[0., 1.], [1., 1.]])
    print(relative_L2_loss(pred, gt))