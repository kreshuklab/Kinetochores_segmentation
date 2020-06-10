# UNet model

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        x = F.elu(x)
        return x

# DownSampling

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 16
        self.num_conv_blocks = 2

        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for num in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                if depth == 0:
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, num)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, num)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        down_sampling_features = []
        
        for op_name, op_call in self.module_dict.items():
            if op_name.startswith("conv"):
                x = op_call(x)
                if op_name.endswith("1"):
                    down_sampling_features.append(x)
            elif op_name.startswith("max_pooling"):
                x = op_call(x)

        return x, down_sampling_features


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)


# UpSampling

class DecoderBlock(nn.Module):
    def __init__(self, out_channels, model_depth=4):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = 16
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for num in range(self.num_conv_blocks):
                if num == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, num)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, num)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, down_sampling_features):
        for op_name, op_call in self.module_dict.items():
            if op_name.startswith("deconv"):
                x = op_call(x)
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
            elif op_name.startswith("conv"):
                x = op_call(x)
            else:
                x = op_call(x)
        return x

class UnetModel(nn.Module):
    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="sigmoid"):
        super(UnetModel, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        x = self.sigmoid(x)
        return x