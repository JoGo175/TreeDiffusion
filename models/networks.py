"""
Encoder, decoder, transformation, router, and dense layer architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm


spectral_normalization = False

def actvn(x):
    return F.silu(x)    # silu = swish
    #return F.relu(x)
    #return F.leaky_relu(x, negative_slope=0.3)


class EncoderSmall(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(EncoderSmall, self).__init__()

        self.dense1 = nn.Linear(in_features=input_shape, out_features=4*output_shape, bias=False)
        self.bn1 = nn.BatchNorm1d(4*output_shape)
        self.dense2 = nn.Linear(in_features=4*output_shape, out_features=4*output_shape, bias=False)
        self.bn2 = nn.BatchNorm1d(4*output_shape)
        self.dense3 = nn.Linear(in_features=4*output_shape, out_features=2*output_shape, bias=False)
        self.bn3 = nn.BatchNorm1d(2*output_shape)
        self.dense4 = nn.Linear(in_features=2*output_shape, out_features=output_shape, bias=False)
        self.bn4 = nn.BatchNorm1d(output_shape)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = actvn(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = actvn(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = actvn(x)
        x = self.dense4(x)
        x = self.bn4(x)
        x = actvn(x)
        return x, None, None

class DecoderSmall(nn.Module):
    def __init__(self, input_shape, output_shape, activation):
        super(DecoderSmall, self).__init__()
        self.activation = activation
        self.dense1 = nn.Linear(in_features=input_shape, out_features=128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.dense2 = nn.Linear(in_features=128, out_features=256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dense3 = nn.Linear(in_features=256, out_features=512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.dense4 = nn.Linear(in_features=512, out_features=512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.dense5 = nn.Linear(in_features=512, out_features=output_shape, bias=True)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = actvn(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = actvn(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = actvn(x)
        x = self.dense4(x)
        x = self.bn4(x)
        x = actvn(x)
        x = self.dense5(x)
        if self.activation == "sigmoid":
            x = torch.sigmoid(x)
        return x


class EncoderSmallCnn(nn.Module):
    def __init__(self, input_shape, input_channels, output_shape, output_channels):
        super(EncoderSmallCnn, self).__init__()
        # interpolate channels and shapes
        channels = np.linspace(input_channels, output_channels, 4, dtype=int)
        shapes = np.linspace(input_shape, output_shape, 4, dtype=int)

        # input_shape -> 4 * output_shape
        s1 = max(int(shapes[0] / (shapes[1])), 1)
        k1 = shapes[0] - s1 * (shapes[1] - 1) + 2 # padding = 1
        self.cnn0 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=k1, stride=s1, padding=1, bias=False)
        # 4 * output_shape -> 2 * output_shape
        s2 = max(int(shapes[1] / (shapes[2])), 1)
        k2 = shapes[1] - s2 * (shapes[2] - 1) + 2 # padding = 1
        self.cnn1 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=k2, stride=s2, padding=1, bias=False)
        # 2 * output_shape -> output_shape
        s3 = max(int(shapes[2] / (shapes[3])), 1)
        k3 = shapes[2] - s3 * (shapes[3] - 1) + 2 # padding = 1
        self.cnn2 = nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=k3, stride=s3, padding=1, bias=False)

        self.bn0 = nn.BatchNorm2d(channels[1])
        self.bn1 = nn.BatchNorm2d(channels[2])
        self.bn2 = nn.BatchNorm2d(channels[3])

        self.spectral_normalization = spectral_normalization
        if self.spectral_normalization:
            self.cnn0 = spectral_norm(self.cnn0)
            self.cnn1 = spectral_norm(self.cnn1)
            self.cnn2 = spectral_norm(self.cnn2)

    def forward(self, x):
        x = self.cnn0(x)
        x = self.bn0(x)
        x = actvn(x)
        x = self.cnn1(x)
        x = self.bn1(x)
        x = actvn(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = actvn(x)
        return x, None, None

class DecoderSmallCnn(nn.Module):
    def __init__(self, input_shape, input_channels, output_shape, output_channels, activation):
        super(DecoderSmallCnn, self).__init__()
        # interpolate channels and shapes
        channels = np.linspace(input_channels, output_channels, 4, dtype=int)
        shapes = np.linspace(input_shape, output_shape, 4, dtype=int)

        # input_shape -> 2 * input_shape
        s1 = max(int(shapes[1] / (shapes[0])), 1)
        k1 = shapes[1] - s1 * (shapes[0] - 1) + 2 # padding = 1
        self.cnn0 = nn.ConvTranspose2d(in_channels=channels[0], out_channels=channels[1], kernel_size=k1, stride=s1, padding=1, bias=False)
        # 2 * input_shape -> 4 * input_shape
        s2 = max(int(shapes[2] / (shapes[1])), 1)
        k2 = shapes[2] - s2 * (shapes[1] - 1) + 2 # padding = 1
        self.cnn1 = nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[2], kernel_size=k2, stride=s2, padding=1, bias=False)
        # 4 * input_shape -> output_shape
        s3 = max(int(shapes[3] / (shapes[2])), 1)
        k3 = shapes[3] - s3 * (shapes[2] - 1) + 2  # padding = 1
        self.cnn2 = nn.ConvTranspose2d(in_channels=channels[2], out_channels=channels[3], kernel_size=k3, stride=s3, padding=1, bias=False)

        self.bn0 = nn.BatchNorm2d(channels[1])
        self.bn1 = nn.BatchNorm2d(channels[2])
        self.activation = activation

        self.spectral_normalization = spectral_normalization
        if self.spectral_normalization:
            self.cnn0 = spectral_norm(self.cnn0)
            self.cnn1 = spectral_norm(self.cnn1)
            self.cnn2 = spectral_norm(self.cnn2)

    def forward(self, inputs):
        x = self.cnn0(inputs)
        x = self.bn0(x)
        x = actvn(x)
        x = self.cnn1(x)
        x = self.bn1(x)
        x = actvn(x)
        x = self.cnn2(x)

        if self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        return x


class EncoderOmniglot(nn.Module):
    def __init__(self, encoded_size):
        super(EncoderOmniglot, self).__init__()
        self.cnns = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=encoded_size//4, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=encoded_size//4, out_channels=encoded_size//4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels=encoded_size//4, out_channels=encoded_size//2, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=encoded_size//2, out_channels=encoded_size//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels=encoded_size//2, out_channels=encoded_size, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=encoded_size, out_channels=encoded_size, kernel_size=5, bias=False)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(encoded_size//4),
            nn.BatchNorm2d(encoded_size//4),
            nn.BatchNorm2d(encoded_size//2),
            nn.BatchNorm2d(encoded_size//2),
            nn.BatchNorm2d(encoded_size),
            nn.BatchNorm2d(encoded_size)
        ])

    def forward(self, x):
        for i in range(len(self.cnns)):
            x = self.cnns[i](x)
            x = self.bns[i](x)
            x = actvn(x)
        x = x.view(x.size(0), -1)
        return x, None, None
        
class DecoderOmniglot(nn.Module):
    def __init__(self, input_shape, activation):
        super(DecoderOmniglot, self).__init__()
        self.activation = activation
        self.dense = nn.Linear(in_features=input_shape, out_features=2 * 2 * 128, bias=False)
        self.cnns = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, bias=False),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=1, bias=False),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=1, bias=False),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=0, output_padding=1, bias=False)
        ])
        self.cnns.append(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=4, stride=1, padding=1, bias=True))
        self.bn = nn.BatchNorm1d(2 * 2 * 128)
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(32)
        ])

    def forward(self, inputs):
        x = self.dense(inputs)
        x = self.bn(x)
        x = actvn(x)
        x = x.view(-1, 128, 2, 2)
        for i in range(len(self.bns)):
            x = self.cnns[i](x)
            x = self.bns[i](x)
            x = actvn(x)
        x = self.cnns[-1](x)
        if self.activation == "sigmoid":
            x = torch.sigmoid(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super(ResnetBlock, self).__init__()

        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(in_channels=fin, out_channels=self.fin, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channels=self.fin, out_channels=self.fhidden, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=self.fhidden, out_channels=self.fout, kernel_size=3, stride=1, padding=1, bias=is_bias)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels=fin, out_channels=self.fout, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(self.fin)
        self.bn1 = nn.BatchNorm2d(self.fin)
        self.bn2 = nn.BatchNorm2d(self.fhidden)

        self.spectral_normalization = spectral_normalization

        if self.spectral_normalization:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            self.conv_2 = spectral_norm(self.conv_2)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)


    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(self.bn0(x)))
        dx = self.conv_1(actvn(self.bn1(dx)))
        dx = self.conv_2(actvn(self.bn2(dx)))
        out = x_s + 0.1 * dx
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

class Resnet_Encoder(nn.Module):
    def __init__(self, input_shape, input_channels, output_shape, output_channels):
        super(Resnet_Encoder, self).__init__()

        nf = 32

        # Submodules
        nlayers = int(torch.log2(torch.tensor(input_shape / output_shape).float()))
        channels = np.linspace(nf, output_channels, nlayers + 1, dtype=int)

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = channels[i]
            nf1 = channels[i + 1]
            blocks += [
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1), # changes spatial size, preserves number of channels
                ResnetBlock(nf0, nf1), # changes number of channels, preserves spatial size
            ]

        self.conv_img = nn.Conv2d(input_channels, nf, kernel_size=3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.bn0 = nn.BatchNorm2d(output_channels)

        self.spectral_normalization = spectral_normalization
        if self.spectral_normalization:
            self.conv_img = spectral_norm(self.conv_img)


    def forward(self, x):
        out = self.conv_img(x)
        out = self.resnet(out)
        out = actvn(self.bn0(out))
        return out, None, None
    
class Resnet_Decoder(nn.Module):
    def __init__(self, input_shape, input_channels, output_shape, output_channels, activation='sigmoid'):
        super(Resnet_Decoder, self).__init__()

        nf = 32 # 128
        #nf_max = 256
        self.activation = activation

        # Submodules
        nlayers = int(torch.log2(torch.tensor(output_shape / input_shape).float()))
        channels = np.linspace(input_channels, nf,  nlayers + 1, dtype=int)

        #self.conv0 = nn.Conv2d(10, nf_max, kernel_size=3, padding=1)

        blocks = []
        for i in range(nlayers):
            nf0 = channels[i]
            nf1 = channels[i + 1]
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ]
        blocks += [
            ResnetBlock(nf, nf),
        ]
        self.resnet = nn.Sequential(*blocks)

        self.bn0 = nn.BatchNorm2d(nf)
        #self.conv_img = nn.ConvTranspose2d(nf, output_channels, kernel_size=3, padding=1)
        self.conv_img = nn.Conv2d(nf, output_channels, kernel_size=3, padding=1)

        self.spectral_normalization = spectral_normalization
        if self.spectral_normalization:
            self.conv_img = spectral_norm(self.conv_img)
            #self.conv0 = spectral_norm(self.conv0)



    def forward(self, z):
        #out = self.conv0(z)
        out = self.resnet(z)
        out = self.conv_img(actvn(self.bn0(out)))
        if self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        return out


class Conv1(nn.Module):
    def __init__(self, input_shape, encoded_size, skip_connection=False):
        super(Conv, self).__init__()
        self.resnet = nn.Sequential(
            ResnetBlock(input_shape, input_shape),
        )

        self.mu = nn.Conv2d(in_channels=input_shape, out_channels=encoded_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigma = nn.Conv2d(in_channels=input_shape, out_channels=encoded_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip_connection = skip_connection

        self.spectral_normalization = spectral_normalization
        if self.spectral_normalization:
            self.mu = spectral_norm(self.mu)
            self.sigma = spectral_norm(self.sigma)
    def forward(self, inputs):
        x = self.resnet(inputs)
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))
        if self.skip_connection:
            if inputs.shape[1] == mu.shape[1]:
                mu = mu + inputs
            else: # throw an error
                ValueError('The skip connection is not possible.')
        return x, mu, sigma

class Conv(nn.Module):
    def __init__(self, input_shape, encoded_size, skip_connection=False):
        super(Conv, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=input_shape, out_channels=input_shape, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(input_shape)
        self.conv1 = nn.Conv2d(in_channels=input_shape, out_channels=input_shape, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_shape)
        self.conv2 = nn.Conv2d(in_channels=input_shape, out_channels=input_shape, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(input_shape)
        self.mu = nn.Conv2d(in_channels=input_shape, out_channels=encoded_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigma = nn.Conv2d(in_channels=input_shape, out_channels=encoded_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip_connection = skip_connection

        self.spectral_normalization = spectral_normalization
        if self.spectral_normalization:
            self.conv0 = spectral_norm(self.conv0)
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.mu = spectral_norm(self.mu)
            self.sigma = spectral_norm(self.sigma)
    def forward(self, inputs):
        x = self.conv0(inputs)
        x = self.bn0(x)
        x = actvn(x) + inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = actvn(x) + inputs
        x = self.conv2(x)
        x = self.bn2(x)
        x = actvn(x) + inputs
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))
        if self.skip_connection:
            if inputs.shape[1] == mu.shape[1]:
                mu = mu + inputs
            else: # throw an error
                ValueError('The skip connection is not possible.')
        return x, mu, sigma

# Small branch transformation
class MLP(nn.Module):
    def __init__(self, input_size, encoded_size, hidden_unit, skip_connection=False):
        super(MLP, self).__init__()
        self.dense1 = nn.Linear(input_size, hidden_unit, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_unit)
        self.mu = nn.Linear(hidden_unit, encoded_size)
        self.sigma = nn.Linear(hidden_unit, encoded_size)
        self.skip_connection = skip_connection
        self.dense2 = nn.Linear(input_size, encoded_size, bias=False)

    def forward(self, inputs):
        x = self.dense1(inputs) 
        x = self.bn1(x)
        x = actvn(x)       
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))

        if self.skip_connection:
            if inputs.shape[1] == mu.shape[1]:
                mu = mu + self.dense2(inputs)
            else:
                mu = mu + self.dense2(inputs)

        return x, mu, sigma


class Dense(nn.Module):
    def __init__(self, input_size, encoded_size):
        super(Dense, self).__init__()
        self.mu = nn.Conv2d(in_channels=input_size, out_channels=encoded_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigma = nn.Conv2d(in_channels=input_size, out_channels=encoded_size, kernel_size=3, stride=1, padding=1, bias=False)

        self.spectral_normalization = spectral_normalization
        if self.spectral_normalization:
            self.mu = spectral_norm(self.mu)
            self.sigma = spectral_norm(self.sigma)

    def forward(self, inputs):
        x = inputs
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))
        return mu, sigma


class Router(nn.Module):
    def __init__(self, input_size, rep_dim, hidden_units=128, dropout=0):
        super(Router, self).__init__()
        # input is (encoded_size, 8, 8)
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.dense1 = nn.Linear(1 * rep_dim * rep_dim, hidden_units, bias=False)
        self.dense2 = nn.Linear(hidden_units, hidden_units, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.dense3 = nn.Linear(hidden_units, 1)
        self.dropout = nn.Dropout(p=dropout)

        self.spectral_normalization = spectral_normalization
        if self.spectral_normalization:
            self.conv1 = spectral_norm(self.conv1)
            self.dense1 = spectral_norm(self.dense1)
            self.dense2 = spectral_norm(self.dense2)
            self.dense3 = spectral_norm(self.dense3)

    def forward(self, inputs, return_last_layer=False):
        x = self.conv1(inputs)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.bn1(x)
        x = actvn(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = actvn(x)
        x = self.dropout(x)
        d = F.sigmoid(self.dense3(x))
        
        if return_last_layer:
            return d, x
        else:
            return d

class contrastive_projection(nn.Module):
    def __init__(self, input_size, rep_dim, hidden_unit=128, encoded_size=10):
        super(contrastive_projection, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.dense1 = nn.Linear(1*rep_dim*rep_dim, hidden_unit, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_unit)
        self.mu = nn.Linear(hidden_unit, encoded_size)
        self.sigma = nn.Linear(hidden_unit, encoded_size)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.bn1(x)
        x = actvn(x)
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))

        return x, mu, sigma
        

def get_encoder(architecture, input_shape, input_channels, output_shape, output_channels):
    if architecture == 'mlp':
        encoder = EncoderSmall(input_shape=input_shape, output_shape=output_shape)
    elif architecture == 'cnn1':
        encoder = EncoderSmallCnn(input_shape, input_channels, output_shape, output_channels)
    elif architecture == 'cnn2':
        encoder = Resnet_Encoder(input_shape, input_channels, output_shape, output_channels)
    elif architecture == 'cnn_omni':
        encoder = EncoderOmniglot(output_shape)
    else:
        raise ValueError('The encoder architecture is mispecified.')
    return encoder


def get_decoder(architecture, input_shape, input_channels, output_shape, output_channels, activation):
    if architecture == 'mlp':
        decoder = DecoderSmall(input_shape, output_shape, activation)
    elif architecture == 'cnn1':
        decoder = DecoderSmallCnn(input_shape, input_channels, output_shape, output_channels, activation)
    elif architecture == 'cnn2':
        decoder = Resnet_Decoder(input_shape, input_channels, output_shape, output_channels, activation)
    elif architecture == 'cnn_omni':
        decoder = DecoderOmniglot(input_shape, activation) 
    else:
        raise ValueError('The decoder architecture is mispecified.')
    return decoder
