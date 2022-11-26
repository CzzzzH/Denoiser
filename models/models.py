import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from ttools.modules.image_operators import crop_like
from . import modules as ops

visual_x = [176, 335, 305]
visual_y = [647, 269, 553]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_kernel(kernel, spp_index, kernel_size=21, mode='sbmc'):

    _, _, h, w = kernel.size()
    kernel = kernel.permute((0, 2, 3, 1))
    kernel = F.softmax(kernel, dim=3).view(-1, h, w, kernel_size, kernel_size)[0].cpu().numpy()

    crop = (kernel_size - 1) // 2
    kernel = kernel[crop:-crop, crop:-crop, :, :]

    for i in range(3):
        k = kernel[visual_y[i], visual_x[i]]
        k = (k - k.min()) / (k.max() - k.min())
        k = cv2.resize(k, (kernel_size * 10, kernel_size * 10))
        cv2.imwrite(f"test_result/sbmc_kernel_{spp_index}_{visual_x[i]}_{visual_y[i]}.png", k * 255)

class SBMC(nn.Module):
    
    """
    Model from Sample-Based Monte-Carlo denoising using a Kernel-Splatting
    Network. [Gharbi2019] <http://groups.csail.mit.edu/graphics/rendernet/>

    Args:
        n_features(int): number of input features per sample, used to predict
            the splatting kernels.
        n_global_features(int): number of global features, also used to
            predict the splats.
        width(int): number of features per conv layer.
        embedding_width(int): number of intermediate per-sample features.
        ksize(int): spatial extent of the splatting kernel (square).
        splat(bool): if True, uses splatting kernel. Otherwise uses gather
            kernels (useful for ablation).
        nsteps(int): number of sample/pixel coordination steps. ("n" in
            Algorithm 1 in the paper)
        pixel(bool): if True, collapses samples by averaging and treat the
            resulting image as a 1spp input. Useful for ablation.
    """

    def __init__(self, n_features, n_global_features, width=128,
                 embedding_width=128, ksize=21, splat=True, nsteps=3,
                 pixel=False):
        super(SBMC, self).__init__()

        if ksize < 3 or (ksize % 2 == 0):
            raise ValueError("Kernel size should be odd and > 3.")

        if nsteps < 1:
            raise ValueError("Multisteps requires at least one sample/pixel "
                             "step.")

        self.ksize = ksize
        self.splat = splat
        self.pixel = pixel
        self.width = width
        self.embedding_width = embedding_width
        self.eps = 1e-8  # for kernel normalization

        # We repeat the pixel/sample alternate processing for `nsteps` steps.
        self.nsteps = nsteps
        for step in range(self.nsteps):
            if step == 0:
                n_in = n_features + n_global_features
            else:
                n_in = self.embedding_width + width

            # 1x1 convolutions implement the per-sample transformation
            self.add_module("embedding_{:02d}".format(step),
                            ops.ConvChain(n_in, self.embedding_width,
                                          width=width, depth=3, ksize=1,
                                          pad=False))

            # U-net implements the pixel spatial propagation step
            self.add_module("propagation_{:02d}".format(step), ops.Autoencoder(
                self.embedding_width, width, num_levels=3, increase_factor=2.0,
                num_convs=3, width=width, ksize=3, output_type="leaky_relu",
                pooling="max"))

        # Final regression for the per-sample kernel (also 1x1 convs)
        self.kernel_regressor = ops.ConvChain(width + self.embedding_width,
                                              ksize*ksize, depth=3,
                                              width=width, ksize=1,
                                              activation="leaky_relu",
                                              pad=False, output_type="linear")

        # This module aggregates the sample contributions
        self.kernel_update = ops.ProgressiveKernelApply(splat=self.splat)

    def forward(self, samples):
        """Forward pass of the model.

        Args:
            data(dict) with keys:
                "radiance": (th.Tensor[bs, spp, 3, h, w]) sample radiance.
                "features": (th.Tensor[bs, spp, nf, h, w]) sample features.
                "global_features": (th.Tensor[bs, ngf, h, w]) global features.

        Returns:
            (dict) with keys:
                "radiance": (th.Tensor[bs, 3, h, w]) denoised radiance
        """
        radiance = samples["radiance"]
        features = samples["features"]
        gfeatures = samples["global_features"]
        
        if self.pixel:
            # Make the pixel-average look like one sample
            radiance = radiance.mean(1, keepdim=True)
            features = features.mean(1, keepdim=True)

        bs, spp, nf, h, w = features.shape

        modules = {n: m for (n, m) in self.named_modules()}

        limit_memory_usage = not self.training

        # -- Embed the samples then collapse to pixel-wise summaries ----------
        if limit_memory_usage:
            gf = gfeatures.repeat([1, 1, h, w])
            new_features = torch.zeros(bs, spp, self.embedding_width, h, w)
        else:
            gf = gfeatures.repeat([spp, 1, h, w])

        for step in range(self.nsteps):
            if limit_memory_usage:
                # Go through the samples one by one to preserve memory for
                # large images
                for sp in range(spp):
                    f = features[:, sp]
                    if step == 0:  # Global features at first iteration only
                        f = torch.cat([f, gf], 1)
                    else:
                        f = torch.cat([f, propagated], 1)

                    f = modules["embedding_{:02d}".format(step)](f)

                    new_features[:, sp].copy_(f, non_blocking=True)

                    if sp == 0:
                        reduced = f
                    else:
                        reduced.add_(f)

                    del f
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                features = new_features.to(device)
                reduced.div_(spp)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                flat = features.view([bs*spp, nf, h, w])
                if step == 0:  # Global features at first iteration only
                    flat = torch.cat([flat, gf], 1)
                else:
                    flat = torch.cat([flat, propagated.unsqueeze(1).repeat(
                        [1, spp, 1, 1, 1]).view(spp*bs, self.width, h, w)], 1)
                flat = modules["embedding_{:02d}".format(step)](flat)
                flat = flat.view(bs, spp, self.embedding_width, h, w)
                reduced = flat.mean(1)
                features = flat
                nf = self.embedding_width

            # Propagate spatially the pixel context
            propagated = modules["propagation_{:02d}".format(step)](reduced)

            if limit_memory_usage:
                del reduced
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Predict kernels based on the context information and
        # the current sample's features
        sum_r, sum_w, max_w = None, None, None

        for sp in range(spp):
            f = features[:, sp].to(radiance.device)
            f = torch.cat([f, propagated], 1)
            r = radiance[:, sp].to(radiance.device)
            kernels = self.kernel_regressor(f)
            if limit_memory_usage:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # save_kernel(kernels, spp_index=sp)

            # Update radiance estimate
            sum_r, sum_w, max_w = self.kernel_update(
                crop_like(r, kernels), kernels, sum_r, sum_w, max_w)
            if limit_memory_usage:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Normalize output with the running sum
        output = sum_r / (sum_w + self.eps)

        # Remove the invalid boundary data
        crop = (self.ksize - 1) // 2
        output = output[..., crop:-crop, crop:-crop]

        return output


class KPCN(nn.Module):
    """
    Model from Kernel-Predicting Convolutional Networks for Denoising Monte Carlo
    Renderings: <http://cvc.ucsb.edu/graphics/Papers/SIGGRAPH2017_KPCN/>

    Args:
        n_in(int): number of input channels in the diffuse/specular streams.
        ksize(int): size of the gather reconstruction kernel.
        depth(int): number of conv layers in each branch.
        width(int): number of feature channels in each branch.
    """
    def __init__(self, n_in, ksize=21, depth=9, width=100):
        super(KPCN, self).__init__()

        self.ksize = ksize
        self.eps = 1e-2
        
        self.diffuse = ops.ConvChain(
            n_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.specular = ops.ConvChain(
            n_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.kernel_apply = ops.KernelApply(softmax=True, splat=False)

    def forward(self, data):
        """Forward pass of the model.

        Args:
            data(dict) with keys:
                "kpcn_diffuse_in":
                "kpcn_specular_in":
                "kpcn_diffuse_buffer":
                "kpcn_specular_buffer":
                "kpcn_albedo":

        Returns:
            (dict) with keys:
                "radiance":
                "diffuse":
                "specular":
        """
        # Process the diffuse and specular channels independently
        k_diffuse = self.diffuse(data["kpcn_diffuse_in"])
        k_specular = self.specular(data["kpcn_specular_in"])

        # Match dimensions
        b_diffuse = crop_like(data["kpcn_diffuse_buffer"],
                              k_diffuse).contiguous()
        b_specular = crop_like(data["kpcn_specular_buffer"],
                               k_specular).contiguous()

        # Kernel reconstruction
        r_diffuse, _ = self.kernel_apply(b_diffuse, k_diffuse)
        r_specular, _ = self.kernel_apply(b_specular, k_specular)

        # Combine diffuse/specular/albedo
        albedo = crop_like(data["kpcn_albedo"], r_diffuse)
        final_specular = torch.exp(r_specular) - 1
        final_diffuse = (albedo + self.eps) * r_diffuse
        final_radiance = final_diffuse + final_specular

        return final_radiance
