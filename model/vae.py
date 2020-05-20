# Copyright 2019 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from torch import nn
from torch.nn import functional as F


class Vae(nn.Module):
    def __init__(self, zsize, layer_count=3, channels=3, last_feature_dim=6):
        super(Vae, self).__init__()

        d = 128
        self.d = d
        self.zsize = zsize
        self.last_feature_dim = last_feature_dim

        self.layer_count = layer_count

        mul = 1
        inputs = channels
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv3d(inputs, d * mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm3d(d * mul))
            inputs = d * mul
            mul *= 2

        self.d_max = inputs

        self.fc1 = nn.Linear(inputs * last_feature_dim * last_feature_dim * last_feature_dim, zsize)
        self.fc2 = nn.Linear(inputs * last_feature_dim * last_feature_dim * last_feature_dim, zsize)

        self.d1 = nn.Linear(zsize, inputs * last_feature_dim * last_feature_dim * last_feature_dim)

        mul = inputs // d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose3d(inputs, d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm3d(d * mul))
            inputs = d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose3d(inputs, channels, 4, 2, 1))

    def encode(self, x):
        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))
            # print(f"encode x:{x.shape}")

        # print(f"x.view:x.shape[0] {x.shape[0]} x self.d_max x self.last_feature_dim^3 {self.d_max} x {self.last_feature_dim}^3")
        x = x.view(x.shape[0], -1)
        # print(f"x.view: {x.shape}")
        h1 = self.fc1(x)
        # print(f"fc: {h1.shape}")
        h2 = self.fc2(x)
        return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        # print(f"decode: {x.shape}")
        x = x.view(x.shape[0], self.zsize)
        # print(f"x.view:{x.shape}")
        x = self.d1(x)
        # print(f"d1; {x.shape}")
        x = x.view(x.shape[0], self.d_max, self.last_feature_dim, self.last_feature_dim, self.last_feature_dim)
        # print(f"x.view: {x.shape}")
        #x = self.deconv1_bn(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)
            # print(f"deconv: {x.shape}")

        x = F.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
        return x

    def forward(self, x):
        # print(f"input x:{x.shape}")
        mu, logvar = self.encode(x)
        # print(f"mu, longvar: {mu.shape}| {logvar.shape}")
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        result = self.decode(z.view(-1, self.zsize, 1, 1))
        result = result.squeeze()
        return result, mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()