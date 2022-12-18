import torch.nn as nn
from .rbbox_head import BBoxHeadRbbox
from ..registry import HEADS
from ..utils import ConvModule
import torch
import math
import matplotlib.pyplot as plt

class BasicResBlock(nn.Module):
    """Basic residual block.
    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.
    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None):
        super(BasicResBlock, self).__init__()

        # main path
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False)
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False)

        # identity path
        self.conv_identity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out


@HEADS.register_module
class ConvFCBBoxHeadRbboxAttn(BBoxHeadRbbox):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHeadRbboxAttn, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        ### 
        # self.conv1 = nn.Conv2d(256,1024,1,1,0)
        # self.relu_1 = nn.ReLU(inplace=False)
        # self.conv2 = nn.Conv2d(1024,1024,1,1,0)
        # self.relu_2 = nn.ReLU(inplace=False)
        self.res_block = BasicResBlock(256,1024)
        self.avg_pool = nn.AvgPool2d(7)
        self.foretoken = nn.Embedding(1,256)

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= (self.roi_feat_size * self.roi_feat_size)

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                # last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
                if isinstance(self.roi_feat_size, int):
                    last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
                elif isinstance(self.roi_feat_size, tuple):
                    assert len(self.roi_feat_size) == 2
                    assert isinstance(self.roi_feat_size[0], int)
                    assert isinstance(self.roi_feat_size[1], int)
                    last_layer_dim *= (self.roi_feat_size[0] * self.roi_feat_size[1])
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHeadRbboxAttn, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        # if self.num_shared_fcs > 0:
        #     if self.with_avg_pool:
        #         x = self.avg_pool(x)
        #     x = x.view(x.size(0), -1)
        #     for fc in self.shared_fcs:
        #         x = self.relu(fc(x))
        # double head method 


        roi_feat_tokens = self.foretoken.weight[:,:,None].repeat(x.shape[0],1,1)
        att_maps = ((torch.bmm(roi_feat_tokens.transpose(1,2),x.view(x.shape[0],x.shape[1],-1)))/math.sqrt(256)).softmax(-1)
        update_tokens = torch.bmm(x.view(x.shape[0],x.shape[1],-1),att_maps.transpose(1,2))

        att_map_refine = ((torch.bmm(update_tokens.transpose(1,2),x.view(x.shape[0],x.shape[1],-1)))/math.sqrt(256)).softmax(-1)
        vis_maps = []
        for i in range(10):
            att_map = att_map_refine[i].detach()
            vis_maps.append(att_map.reshape(7,7))
        fig1 = plt.figure(figsize=(20,20))
        for i in range(10):
            fig1.add_subplot(2,5,i+1)
            plt.imshow(vis_maps[i].cpu().numpy())
            plt.axis('off')
        plt.savefig('attn_map.jpg')



        # w = math.sqrt(x.shape[2])
        x = (att_map_refine*x.view(x.shape[0],x.shape[1],-1)).reshape(x.shape[0],x.shape[1],x.shape[2],x.shape[3])

        cls_feat = x # 1024*256*7*7
        reg_feat = x


        cls_feat = cls_feat.view(cls_feat.size(0),-1)
        for fc in self.shared_fcs:
            cls_feat = self.relu(fc(cls_feat))
        cls_score = self.fc_cls(cls_feat) if self.with_cls else None

        # reg_feat = self.relu_1(self.conv1(reg_feat))
        # reg_feat = self.relu_2(self.conv2(reg_feat))
        for i in range(1):
            reg_feat = self.res_block(reg_feat)
        reg_feat = self.avg_pool(reg_feat).reshape(reg_feat.shape[0],reg_feat.shape[1])
        bbox_pred = self.fc_reg(reg_feat) if self.with_reg else None

        # # separate branches
        # x_cls = x
        # x_reg = x

        # for conv in self.cls_convs:
        #     x_cls = conv(x_cls)
        # if x_cls.dim() > 2:
        #     if self.with_avg_pool:
        #         x_cls = self.avg_pool(x_cls)
        #     x_cls = x_cls.view(x_cls.size(0), -1)
        # for fc in self.cls_fcs:
        #     x_cls = self.relu(fc(x_cls))

        # for conv in self.reg_convs:
        #     x_reg = conv(x_reg)
        # if x_reg.dim() > 2:
        #     if self.with_avg_pool:
        #         x_reg = self.avg_pool(x_reg)
        #     x_reg = x_reg.view(x_reg.size(0), -1)
        # for fc in self.reg_fcs:
        #     x_reg = self.relu(fc(x_reg))

        # cls_score = self.fc_cls(x_cls) if self.with_cls else None
        # bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred


@HEADS.register_module
class SharedFCBBoxHeadRbboxAttn(ConvFCBBoxHeadRbboxAttn):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHeadRbboxAttn, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
