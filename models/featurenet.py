import torch.nn as nn
import torch
from models.utils import conv_layer, de_conv_layer


class AutoEncoder(nn.Module):
    def __init__(self, in_channel=1, latent_dim=30, vae=False):
        super(AutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.vae = vae
        self.encoder = FeatureExtractorNet(in_channel)
        if vae:
            self.flatten = nn.Flatten()
            self.linear_mu = nn.Sequential(
                nn.Linear(256 * 32 * 128, latent_dim),
                nn.Tanh()
            )
            self.linear_logvar = nn.Sequential(
                nn.Linear(256 * 32 * 128, latent_dim),
                nn.Tanh()
            )
            self.reverse_linear = nn.Linear(latent_dim, 256 * 32 * 128)
            # self.conv1 = nn.Conv2d(256, latent_dim, 16, stride=1, padding=0)
            # self.conv2 = nn.Conv2d(256, latent_dim, 16, stride=1, padding=0)
            # self.conv3 = nn.ConvTranspose2d(latent_dim, 256, 1, stride=1, padding=0)
        self.decoder = Decoder(in_channel)

    def bottleneck(self, x):
        # mu, log_var = self.conv1(x), self.conv2(x)
        x = x.view(x.size(0), -1)
        mu, log_var = self.linear_mu(x), self.linear_logvar(x)
        z = self.re_parameterize(mu, log_var)
        return z, mu, log_var

    @staticmethod
    def re_parameterize(mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        return z

    def forward(self, x):
        mu = 0
        log_var = 0
        for layers in self.encoder.encoder:
            x = layers(x)
        if self.vae:
            x, mu, log_var = self.bottleneck(x)
            x = self.reverse_linear(x)
            x = x.view(x.size(0), 256, 32, 128)
        for layers in self.decoder.decoder:
            x = layers(x)
        return x, mu, log_var


class FeatureExtractorNet(nn.Module):
    def __init__(self, in_channel=1):
        super(FeatureExtractorNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(conv_layer("fe_l1", in_channel, 2))  # [1, 16, 16, 512]
        self.encoder.append(conv_layer("fe_l2", 2, 4, max_pooling=False))  # [1, 32, 16, 512]
        self.encoder.append(conv_layer("fe_l3", 4, 8))  # [1, 64, 8, 256]
        self.encoder.append(conv_layer("fe_l4", 8, 16, max_pooling=False))  # [1, 128, 8, 256]
        self.encoder.append(conv_layer("fe_l5", 16, 32))  # [1, 256, 4, 128]


class Decoder(nn.Module):
    def __init__(self, in_channel=1):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList()
        self.decoder.append(de_conv_layer("de_fe_l1", 32, 16, decoder=True))  # [1, 16, 16, 512]
        self.decoder.append(de_conv_layer("de_fe_l2", 16, 8, max_pooled=False, decoder=True))  # [1, 32, 16, 512]
        self.decoder.append(de_conv_layer("de_fe_l3", 8, 4, decoder=True))  # [1, 64, 8, 256]
        self.decoder.append(de_conv_layer("de_fe_l4", 4, 2, max_pooled=False, decoder=True))  # [1, 128, 8, 256]
        self.decoder.append(de_conv_layer("de_fe_l5", 2, in_channel, decoder=True))  # [1, 256, 4, 128]
