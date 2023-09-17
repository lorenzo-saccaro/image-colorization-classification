import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class Classifier(nn.Module):

    def __init__(self, n_classes=205, freeze_features=False):
        """
        Classifier
        :param n_classes: number of classes
        :param freeze_features: whether to freeze features extractor
        """
        super().__init__()

        self.classifier = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        # change last layer
        self.classifier.classifier[-1] = nn.Linear(self.classifier.classifier[-1].in_features,
                                                   n_classes)
        # init last layer
        nn.init.xavier_uniform_(self.classifier.classifier[-1].weight)
        nn.init.zeros_(self.classifier.classifier[-1].bias)

        # freeze features extractor if needed
        if freeze_features:
            for param in self.classifier.features.parameters():
                param.requires_grad = False

        self.preprocess = EfficientNet_B3_Weights.IMAGENET1K_V1.transforms(antialias=False)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.classifier(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=4, padding=1, stride=2, batch_norm=True,
                 leaky_relu_slope=0.2, use_instance_norm=False):
        """
        Convolutional block
        :param in_size: input depth
        :param out_size: output depth
        :param kernel_size: kernel size
        :param padding: padding
        :param stride: stride
        :param batch_norm: whether to use batch normalization/instance normalization
        :param leaky_relu_slope: slope of leaky ReLU
        :param use_instance_norm: use instance normalization if batch size is 1
        """

        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride,
                      bias=False),
            nn.BatchNorm2d(out_size) if (batch_norm and not use_instance_norm) else None,
            nn.InstanceNorm2d(out_size, affine=True) if (
                    batch_norm and use_instance_norm) else None,
            nn.LeakyReLU(inplace=True, negative_slope=leaky_relu_slope))

        # remove None from conv_block
        self.conv_block = nn.Sequential(*[x for x in self.conv_block if x is not None])

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(module.weight.data, 1.0, 0.2)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        y = self.conv_block(x)

        return y


class PixelTCL(nn.Module):
    def __init__(self, in_size, out_size):
        """
        Pixel Transposed Convolution Layer as described by Gao, Yuan, Wang, Ji
        :param in_size: number of input channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)

        self.mask3 = torch.tensor([[[[0, 1, 0],
                                     [1, 0, 1],
                                     [0, 1, 0]]]])
        if torch.cuda.is_available(): self.mask3 = self.mask3.to('cuda')
        # self.mask3 = self.mask3.repeat_interleave(3, dim=2)

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def _inflate(self, x, mask_type):
        """
        Add blank rows and columns to create the correct tensor's shape for the convolution
        """
        inflated = x.repeat_interleave(2, dim=2)
        inflated = inflated.repeat_interleave(2, dim=3)

        if mask_type == 'a':
            inflated[:, :, 1::2, :] = 0
            inflated[:, :, :, 1::2] = 0
        elif mask_type == 'b':
            inflated[:, :, ::2, :] = 0
            inflated[:, :, :, ::2] = 0
        else:
            raise 'Wrong mask type, should be \'a\' or \'b\''

        return inflated

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        a_inflated = self._inflate(a, mask_type='a')
        b_inflated = self._inflate(b, mask_type='b')
        c = a_inflated + b_inflated
        self.conv3.weight.data *= self.mask3
        d = self.conv3(c)

        return c + d


class DecoderBlock(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=4, padding=1, stride=2, batch_norm=True,
                 apply_dropout=True, dropout_p=0.5, use_instance_norm=False, use_pixelTCL=False):
        """
        Convolutional block
        :param in_size: input depth
        :param out_size: output depth
        :param kernel_size: kernel size
        :param padding: padding
        :param stride: stride
        :param batch_norm: whether to use batch normalization
        :param apply_dropout: whether to apply dropout
        :param dropout_p: dropout probability
        :param use_instance_norm: use instance normalization if batch size is 1
        :param use_pixelTCL: wheter to use PixelTCL instead of ConvTranspose2d (for pixelCNN)
        """

        super().__init__()

        if not use_pixelTCL:
            self.decoder_block = nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, padding=padding,
                                   stride=stride, bias=False),
                nn.BatchNorm2d(out_size) if (batch_norm and not use_instance_norm) else None,
                nn.InstanceNorm2d(out_size, affine=True) if (
                        batch_norm and use_instance_norm) else None,
                nn.Dropout2d(p=dropout_p) if apply_dropout else None,
                nn.ReLU(inplace=True))

            # remove None from decoder_block
            self.decoder_block = nn.Sequential(
                *[x for x in self.decoder_block if x is not None])
        else:
            self.decoder_block = nn.Sequential(
                PixelTCL(in_size, out_size),
                nn.BatchNorm2d(out_size) if (batch_norm and not use_instance_norm) else None,
                nn.InstanceNorm2d(out_size, affine=True) if (
                        batch_norm and use_instance_norm) else None,
                nn.Dropout2d(p=dropout_p) if apply_dropout else None,
                nn.ReLU(inplace=True))
            # remove None from decoder_block
            self.decoder_block = nn.Sequential(
                *[x for x in self.decoder_block if x is not None])

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(module.weight.data, 1.0, 0.2)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        y = self.decoder_block(x)

        return y


class Generator(nn.Module):

    def __init__(self, in_channels=1, out_channels=3, filters=(
            64, 128, 256, 512, 512, 512, 512, 512), use_instance_norm=False,
                 leaky_relu_slope=0.2, use_pixelTCL=False, classifier_branch=False,
                 num_classes=205):
        """
        Generator network
        :param in_channels: input channels, 1 for grayscale
        :param out_channels: output channels, 3 for RGB
        :param filters: number of filters in each layer
        :param use_instance_norm: use instance normalization if batch size is 1
        :param leaky_relu_slope: slope of leaky ReLU
        :param use_pixelTCL: whether to use PixelTCL in decoder instead (for pixelCNN)
        :param classifier_branch: whether to add a classifier branch
        :param num_classes: number of classes
        """
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.class_branch = None
        self.feature_fusion = None
        self.class_head = None
        if classifier_branch:
            self.class_branch = nn.ModuleList()
            # TODO: try changing number of neurons in linear layers
            self.feature_fusion = nn.Sequential(nn.LazyLinear(1024),
                                                nn.LeakyReLU(leaky_relu_slope),
                                                nn.LazyLinear(512),
                                                nn.LeakyReLU(leaky_relu_slope))
            self.class_head = nn.Sequential(nn.LazyLinear(2048),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.LazyLinear(1024),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.LazyLinear(num_classes))

        self.last = nn.ConvTranspose2d(2 * filters[0], out_channels, kernel_size=4, padding=1,
                                       stride=2)
        # initialize weights for last layer using xavier uniform
        nn.init.xavier_uniform_(self.last.weight)
        nn.init.constant_(self.last.bias, 0)

        self.final_activation = nn.Tanh()

        # Encoder
        for i, n_filters in enumerate(filters):
            if classifier_branch and  2 < i < 6:
                self.class_branch.append(EncoderBlock(filters[i - 1], n_filters,
                                                    use_instance_norm=use_instance_norm,
                                                    leaky_relu_slope=leaky_relu_slope))

            if i == 0:
                self.encoder.append(EncoderBlock(in_channels, n_filters, batch_norm=False,
                                                 use_instance_norm=use_instance_norm,
                                                 leaky_relu_slope=leaky_relu_slope))
            # cannot use instance norm for last layer of encoder because spatial dimensions are 1
            elif i == len(filters) - 1:
                self.encoder.append(EncoderBlock(filters[i - 1], n_filters,
                                                 batch_norm=(not use_instance_norm),
                                                 use_instance_norm=use_instance_norm,
                                                 leaky_relu_slope=leaky_relu_slope))
            else:
                self.encoder.append(EncoderBlock(filters[i - 1], n_filters,
                                                 use_instance_norm=use_instance_norm,
                                                 leaky_relu_slope=leaky_relu_slope))
        if classifier_branch:
            self.class_branch.append(nn.Flatten())

        # Decoder
        for i, n_filters in enumerate(reversed(filters[:-1])):
            if i == 0:
                self.decoder.append(DecoderBlock(filters[-i - 1] * int((1 + classifier_branch)),
                                                 n_filters, apply_dropout=True,
                                                 use_instance_norm=use_instance_norm,
                                                 use_pixelTCL=use_pixelTCL))
            elif 1 <= i <= 2:
                self.decoder.append(DecoderBlock(2 * filters[-i - 1], n_filters, apply_dropout=True,
                                                 use_instance_norm=use_instance_norm,
                                                 use_pixelTCL=use_pixelTCL))
            else:
                self.decoder.append(DecoderBlock(2 * filters[-i - 1], n_filters, apply_dropout=False,
                                                 use_instance_norm=use_instance_norm,
                                                 use_pixelTCL=use_pixelTCL))

    def forward(self, x, return_class_out=True):

        # Create a list to store skip-connections
        skip = []

        # Encoder:
        for i, encoder_step in enumerate(self.encoder):
            if i == 3 and self.class_branch is not None:
                class_branch_in = x # save input to classifier branch
            x = encoder_step(x)
            skip.append(x)

        # Classifier branch
        if self.class_branch is not None:
            class_branch_out = self.class_branch[0](class_branch_in)
            for layer in self.class_branch[1:]:
                class_branch_out = layer(class_branch_out)

        # Feature fusion
        if self.feature_fusion is not None:
            feature_fusion_out = self.feature_fusion(class_branch_out)
            # reshape and concatenate with last encoder output
            feature_fusion_out = feature_fusion_out.unsqueeze(-1).unsqueeze(-1)
            feature_fusion_out = feature_fusion_out.expand_as(x)
            x = torch.cat((x, feature_fusion_out), dim=1)

        # Classification head
        if self.class_head is not None:
            class_logits = self.class_head(class_branch_out)

        skip = skip[:-1]  # remove last element

        # Decoder:
        for skip_connection, decoder_step in zip(reversed(skip), self.decoder):
            x = decoder_step(x)
            x = torch.cat((x, skip_connection), dim=1)

        # Last layer:
        x = self.last(x)
        x = self.final_activation(x)

        if self.class_head is not None and return_class_out:
            return x, class_logits
        elif self.class_head is None and return_class_out:
            return x, None
        else:
            return x


class Discriminator(nn.Module):
    def __init__(self, leaky_relu_slope=0.2, in_channels=4, filters=(64, 128, 256, 512),
                 use_instance_norm=False, disable_norm=False, network_type='patch_gan'):
        """
        Discriminator network
        :param leaky_relu_slope: slope of leaky ReLU
        :param in_channels: input channels, 4 for RGB + grayscale
        :param filters: number of filters in each layer
        :param use_instance_norm: use instance normalization if batch size is 1
        :param disable_norm: disable normalization (for WGAN-GP)
        :param network_type: type of discriminator network, 'patch_gan' or 'DCGAN'
        """
        super(Discriminator, self).__init__()

        self.leaky_relu_slope = leaky_relu_slope
        self.in_channels = in_channels
        self.filters = filters
        assert network_type in ['patch_gan',
                                'DCGAN'], 'network_type must be either patch_gan or DCGAN'
        self.network_type = network_type

        self.model = nn.ModuleList()

        for i, n_filters in enumerate(filters):
            if i == 0:
                self.model.append(EncoderBlock(in_channels, n_filters, kernel_size=4,
                                               batch_norm=False,
                                               leaky_relu_slope=leaky_relu_slope,
                                               use_instance_norm=use_instance_norm))
            elif 1 <= i <= 2:
                self.model.append(EncoderBlock(filters[i - 1], n_filters, kernel_size=4,
                                               batch_norm=(not disable_norm),
                                               leaky_relu_slope=leaky_relu_slope,
                                               use_instance_norm=use_instance_norm))
            else:
                self.model.append(EncoderBlock(filters[i - 1], n_filters, kernel_size=4,
                                               batch_norm=(not disable_norm),
                                               stride=1, leaky_relu_slope=leaky_relu_slope,
                                               use_instance_norm=use_instance_norm))

        if self.network_type == 'patch_gan':
            last = nn.Conv2d(filters[-1], 1, kernel_size=4, stride=1, padding=1)
            # initialize weights for last layer using xavier uniform
            nn.init.xavier_uniform_(last.weight)
            nn.init.constant_(last.bias, 0)
            self.model.append(last)
        elif self.network_type == 'DCGAN':
            self.model.append(nn.AdaptiveAvgPool2d(1))
            self.model.append(nn.Flatten())
            self.model.append(nn.LazyLinear(1))

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


if __name__ == '__main__':
    # Test the network
    from torchinfo import summary

    x = torch.randn(1, 1, 256, 256, device='cuda')
    y = torch.randn(1, 3, 256, 256, device='cuda')
    use_instance_norm = True if x.shape[0] == 1 else False
    gen = Generator(use_instance_norm=use_instance_norm).to('cuda')
    disc = Discriminator(use_instance_norm=use_instance_norm).to('cuda')
    summary(gen, depth=4, input_data=x)
    summary(disc, depth=4, input_data=torch.cat((x, y), dim=1))
    y = gen(x)
    print(y.shape)
    z = disc(torch.cat([x, y], dim=1))
    print(z.shape)
