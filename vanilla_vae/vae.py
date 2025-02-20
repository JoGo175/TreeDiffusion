"""
This file has been modified from a file in the original DiffuseVAE reporitory
which was released under the MIT License, to adapt and improve it for the TreeVAE project.

Source:
https://github.com/kpandey008/DiffuseVAE?tab=readme-ov-file

---------------------------------------------------------------
MIT License

Copyright (c) 2021 Kushagra Pandey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---------------------------------------------------------------
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def parse_layer_string(s):
    layers = []
    for ss in s.split(","):
        if "x" in ss:
            # Denotes a block repetition operation
            res, num = ss.split("x")
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif "u" in ss:
            # Denotes a resolution upsampling operation
            res, mixin = [int(a) for a in ss.split("u")]
            layers.append((res, mixin))
        elif "d" in ss:
            # Denotes a resolution downsampling operation
            res, down_rate = [int(a) for a in ss.split("d")]
            layers.append((res, down_rate))
        elif "t" in ss:
            # Denotes a resolution transition operation
            res1, res2 = [int(a) for a in ss.split("t")]
            layers.append(((res1, res2), None))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def parse_channel_string(s):
    channel_config = {}
    for ss in s.split(","):
        res, in_channels = ss.split(":")
        channel_config[int(res)] = int(in_channels)
    return channel_config


def get_conv(
    in_dim,
    out_dim,
    kernel_size,
    stride,
    padding,
    zero_bias=True,
    zero_weights=False,
    groups=1,
):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_width,
        middle_width,
        out_width,
        down_rate=None,
        residual=False,
        use_3x3=True,
        zero_last=False,
    ):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c3 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


class Encoder(nn.Module):
    def __init__(self, input_res, input_channels, block_config_str, channel_config_str):
        super().__init__()

        self.input_res = input_res

        # If image resolution is not a power of two, add padding
        # Important for the model to work for MNIST, FashionMNIST, etc
        if not (input_res > 0 and (input_res & (input_res - 1)) == 0):
            self.new_input_res = 2 ** int(np.ceil(np.log2(input_res)))
        else:
            self.new_input_res = input_res

        block_config = parse_layer_string(block_config_str)
        channel_config = parse_channel_string(channel_config_str)

        # Initial Convolution
        self.in_conv = nn.Conv2d(input_channels, channel_config[self.new_input_res], 3, stride=1, padding=1, bias=False)

        blocks = []
        for _, (res, down_rate) in enumerate(block_config):
            if isinstance(res, tuple):
                # Denotes transition to another resolution
                res1, res2 = res
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
                )
                continue
            in_channel = channel_config[res]
            use_3x3 = res > 1
            blocks.append(
                ResBlock(
                    in_channel,
                    int(0.5 * in_channel),
                    in_channel,
                    down_rate=down_rate,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        self.block_mod = nn.Sequential(*blocks)

        # Latents
        latent_res = block_config[-1][0]
        self.mu = nn.Conv2d(channel_config[latent_res], channel_config[latent_res], 1, bias=False)
        self.logvar = nn.Conv2d(channel_config[latent_res], channel_config[latent_res], 1, bias=False)

    def forward(self, input):
        # If image resolution not in 32, 64, 128 --> Add padding
        # Important for the model to work for MNIST, FashionMNIST, etc
        if self.input_res != self.new_input_res:
            pad = (self.new_input_res - self.input_res) // 2
            input = F.pad(input, (pad, pad, pad, pad), mode="constant", value=0)

        x = self.in_conv(input)
        x = self.block_mod(x)
        return self.mu(x), self.logvar(x)


class Decoder(nn.Module):
    def __init__(self, input_res, input_channels, block_config_str, channel_config_str):
        super().__init__()

        self.input_res = input_res

        # If image resolution is not a power of two, add padding
        # Important for the model to work for MNIST, FashionMNIST, etc
        if not (input_res > 0 and (input_res & (input_res - 1)) == 0):
            self.new_input_res = 2 ** int(np.ceil(np.log2(input_res)))
        else:
            self.new_input_res = input_res

        block_config = parse_layer_string(block_config_str)
        channel_config = parse_channel_string(channel_config_str)
        blocks = []
        for _, (res, up_rate) in enumerate(block_config):
            if isinstance(res, tuple):
                # Denotes transition to another resolution
                res1, res2 = res
                blocks.append(
                    nn.Conv2d(channel_config[res1], channel_config[res2], 1, bias=False)
                )
                continue

            if up_rate is not None:
                blocks.append(nn.Upsample(scale_factor=up_rate, mode="nearest"))
                continue

            in_channel = channel_config[res]
            use_3x3 = res > 1
            blocks.append(
                ResBlock(
                    in_channel,
                    int(0.5 * in_channel),
                    in_channel,
                    down_rate=None,
                    residual=True,
                    use_3x3=use_3x3,
                )
            )
        # TODO: If the training is unstable try using scaling the weights
        self.block_mod = nn.Sequential(*blocks)
        self.last_conv = nn.Conv2d(channel_config[self.new_input_res ], input_channels, 3, stride=1, padding=1)

    def forward(self, input):
        x = self.block_mod(input)
        x = self.last_conv(x)

        # If image resolution not in 32, 64, 128 --> Remove padding
        # Important for the model to work for MNIST, FashionMNIST, etc
        if self.input_res != self.new_input_res:
            pad = (self.new_input_res - self.input_res ) // 2
            x = x[:, :, pad:-pad, pad:-pad]

        return torch.sigmoid(x)


# Implementation of the Resnet-VAE using a ResNet backbone as encoder
# and Upsampling blocks as the decoder
class VAE(pl.LightningModule):
    def __init__(
        self,
        input_res,
        input_channels,
        enc_block_str,
        dec_block_str,
        enc_channel_str,
        dec_channel_str,
        alpha=1.0,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_res = input_res
        self.input_channels = input_channels
        self.enc_block_str = enc_block_str
        self.dec_block_str = dec_block_str
        self.enc_channel_str = enc_channel_str
        self.dec_channel_str = dec_channel_str
        self.alpha = alpha
        self.lr = lr

        # Encoder architecture
        self.enc = Encoder(self.input_res, self.input_channels, self.enc_block_str, self.enc_channel_str)

        # Decoder Architecture
        self.dec = Decoder(self.input_res, self.input_channels, self.dec_block_str, self.dec_channel_str)

    def encode(self, x):
        mu, logvar = self.enc(x)
        return mu, logvar

    def decode(self, z):
        return self.dec(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, z):
        # Only sample during inference
        decoder_out = self.decode(z)
        return decoder_out

    def forward_recons(self, x):
        # For generating reconstructions during inference
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z)
        return decoder_out

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Encoder
        mu, logvar = self.encode(x)

        # Reparameterization Trick
        z = self.reparameterize(mu, logvar)

        # Decoder
        decoder_out = self.decode(z)

        # Compute loss
        mse_loss = nn.MSELoss(reduction="sum")
        recons_loss = mse_loss(decoder_out, x)
        kl_loss = self.compute_kl(mu, logvar)
        self.log("Recons Loss", recons_loss, prog_bar=True)
        self.log("Kl Loss", kl_loss, prog_bar=True)

        total_loss = recons_loss + self.alpha * kl_loss
        self.log("Total Loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    enc_block_config_str = "128x1,128d2,128t64,64x3,64d2,64t32,32x3,32d2,32t16,16x7,16d2,16t8,8x3,8d2,8t4,4x3,4d4,4t1,1x2"
    enc_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"

    dec_block_config_str = "1x1,1u4,1t4,4x2,4u2,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
    dec_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"

    vae = VAE(
        enc_block_config_str,
        dec_block_config_str,
        enc_channel_config_str,
        dec_channel_config_str,
    )

    sample = torch.randn(1, 3, 128, 128)
    out = vae.training_step(sample, 0)
    print(vae)
    print(out.shape)
