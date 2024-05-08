import MinkowskiEngine.MinkowskiOps as me
import MinkowskiEngine as ME
from MinkowskiEngine import MinkowskiReLU, MinkowskiNetwork, MinkowskiGlobalAvgPooling, MinkowskiAvgPooling
from icg_net.minkowski.modules.common import ConvType, NormType, conv, conv_tr, get_norm, sum_pool
from icg_net.minkowski.modules.resnet_block import BasicBlock, Bottleneck
import torch.nn as nn
import torch


def array2vector(array, step):
    array, step = array.long(), step.long()
    vector = sum([array[:, i] * (step**i) for i in range(1, array.shape[-1])]) + (
        array[:, 0] * step ** array.shape[-1]
    )
    return vector


def sort_spare_tensor(sparse_tensor):
    # https://github.com/NVIDIA/MinkowskiEngine/issues/554
    indices_sort = torch.argsort(array2vector(sparse_tensor.C, sparse_tensor.C.max() + 1))

    sparse_tensor_sort = ME.SparseTensor(
        features=sparse_tensor.F[indices_sort],
        coordinates=sparse_tensor.C[indices_sort],
        tensor_stride=sparse_tensor.tensor_stride[0],
        device=sparse_tensor.device,
    )
    return sparse_tensor_sort, indices_sort


class Model(MinkowskiNetwork):
    """
    Base network for all sparse convnet

    By default, all networks are segmentation networks.
    """

    OUT_PIXEL_DIST = -1

    def __init__(self, in_channels, out_channels, config, D, **kwargs):
        super().__init__(D)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = config


class ResNetBase(Model):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)
    OUT_PIXEL_DIST = 32
    HAS_LAST_BLOCK = False
    CONV_TYPE = ConvType.HYPERCUBE

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        assert self.BLOCK is not None
        assert self.OUT_PIXEL_DIST > 0

        super().__init__(in_channels, out_channels, config, D, **kwargs)

        self.network_initialization(in_channels, out_channels, config, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, config, D):
        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        dilations = config.dilations
        bn_momentum = config.bn_momentum
        self.inplanes = self.INIT_DIM
        self.conv1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
            stride=1,
            D=D,
        )

        self.bn1 = get_norm(NormType.BATCH_NORM, self.inplanes, D=self.D, bn_momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = sum_pool(kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), D=D)

        self.layer1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[0], 1),
        )
        self.layer2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[1], 1),
        )
        self.layer3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[2], 1),
        )
        self.layer4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[3], 1),
        )

        self.final = conv(
            self.PLANES[3] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            D=D,
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilation=1,
        norm_type=NormType.BATCH_NORM,
        bn_momentum=0.1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    D=self.D,
                ),
                get_norm(
                    norm_type,
                    planes * block.expansion,
                    D=self.D,
                    bn_momentum=bn_momentum,
                ),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_type=self.CONV_TYPE,
                D=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    conv_type=self.CONV_TYPE,
                    D=self.D,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final(x)
        return x


class Res16UNetBase(ResNetBase):
    BLOCK = None
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    DILATIONS = (1, 2, 2, 2, 2, 2, 2, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self, in_channels, out_channels, config, D=3, out_fpn=False, **kwargs):
        super().__init__(in_channels, out_channels, config, D)
        self.out_fpn = out_fpn
        self.glob_pool = MinkowskiGlobalAvgPooling()
        self.pool2 = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

    def network_initialization(self, in_channels, out_channels, config, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = config.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )

        self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

        self.conv1p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv2p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv3p4s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv4p8s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr4p16s2 = conv_tr(
            self.inplanes,
            self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr5p8s2 = conv_tr(
            self.inplanes,
            self.PLANES[5],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr5 = get_norm(self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr6p4s2 = conv_tr(
            self.inplanes,
            self.PLANES[6],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr6 = get_norm(self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(
            self.BLOCK,
            self.PLANES[6],
            self.LAYERS[6],
            dilation=dilations[6],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr7p2s2 = conv_tr(
            self.inplanes,
            self.PLANES[7],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr7 = get_norm(self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            dilation=dilations[7],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.final = conv(self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D)
        self.relu = MinkowskiReLU(inplace=True)

    def forward(self, x, repr_sort_idxs=False):
        feature_maps = []

        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # pixel_dist=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        feature_maps.append(out)

        # pixel_dist=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Ugly pooling tries
        # p1 = self.pool2(out)
        # p2 = self.pool2(p1)
        # avg_pool = self.glob_pool(p2)
        # inter_steps = [avg_pool, p2, p2]
        # feature_maps.append(avg_pool)
        # feature_maps.append(p2)
        # feature_maps.append(p1)
        # feature_maps.append(out)

        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        feature_maps.append(out)

        # pixel_dist=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        feature_maps.append(out)

        # pixel_dist=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        feature_maps.append(out)

        # pixel_dist=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = me.cat(out, out_p1)
        out = self.block8(out)

        feature_maps.append(out)

        if not repr_sort_idxs:
            if not self.out_fpn:
                return out
            else:
                # inter_steps.extend(feature_maps)
                return out, feature_maps  # , inter_steps
        else:
            if not self.out_fpn:
                return sort_spare_tensor(out)
            else:
                # inter_steps.extend(feature_maps)
                return sort_spare_tensor(out), [sort_spare_tensor(m) for m in feature_maps]  #


class Res16UNet14(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16UNet18(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class Res16UNet34(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet50(Res16UNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet101(Res16UNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class Res16UNet14A(Res16UNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNetTiny(Res16UNet14):
    INIT_DIM = 8
    PLANES = (16, 32, 64, 128, 64, 64, 32, 32)


class Res16UNetSmall(Res16UNet14):
    INIT_DIM = 8
    PLANES = (32, 64, 128, 128, 96, 96, 64, 32)


class Res16UNet14A2(Res16UNet14A):
    LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B(Res16UNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet14B2(Res16UNet14B):
    LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B3(Res16UNet14B):
    LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16UNet14C(Res16UNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16UNet14D(Res16UNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18A(Res16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18B(Res16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet18D(Res16UNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet34A(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet34B(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34C(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class Custom30M(Res16UNet34):
    PLANES = (32, 64, 128, 256, 128, 64, 64, 32)


class Res16UNet34D(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 128)


class STRes16UNetBase(Res16UNetBase):
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
        super().__init__(in_channels, out_channels, config, D, **kwargs)


class STRes16UNet14(STRes16UNetBase, Res16UNet14):
    pass


class STRes16UNet14A(STRes16UNetBase, Res16UNet14A):
    pass


class STRes16UNet18(STRes16UNetBase, Res16UNet18):
    pass


class STRes16UNet34(STRes16UNetBase, Res16UNet34):
    pass


class STRes16UNet50(STRes16UNetBase, Res16UNet50):
    pass


class STRes16UNet101(STRes16UNetBase, Res16UNet101):
    pass


class STRes16UNet18A(STRes16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class STResTesseract16UNetBase(STRes16UNetBase):
    pass
    # CONV_TYPE = ConvType.HYPERCUBE


class STResTesseract16UNet18A(STRes16UNet18A, STResTesseract16UNetBase):
    pass
