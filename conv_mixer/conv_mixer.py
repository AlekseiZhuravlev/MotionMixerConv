import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiChanSELayer(nn.Module):
    """
        Squeeze-and-Excitation layer for Tensors of Shape [bs, nChan, nTP, dimPosEmb].
        If nChan = 1, this is equivalent to the original SE-Layer. 
        Valentin: I am not 100% satisfied with the MotionMixer-Paper at this point: 
        SE-Architectures were originally designed to weight a bunch of channels 
        and not a bunch of time points.
        TODO: In later experiments, we could set our number of channels (nChan) to
            a value > 1 and apply the SE-Layer as planned by the original SE-paper 
            authors: I.e.  [bs, nChan, nTP, dimPosEmb] -(squeeze)-> [bs, nChan] 
            -> [bs, nChan // r] -> [bs, nChan] -> [bs, nChan, 1, 1] -> [bs, nChan, nTP, dimPosEmb]
    """
    def __init__(self, nTP, r=4, use_max_pooling=False):
        super().__init__()
        self.squeezeBlock = nn.AdaptiveAvgPool2d((1,1)) if not use_max_pooling else nn.AdaptiveMaxPool2d((1,1))
        self.excitationBlock = nn.Sequential(
            nn.Linear(nTP, nTP // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nTP // r, nTP, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x): # Input: [bs, nChan, nTP, dimPosEmb]
        bs, _, nTP, _ = x.shape # [bs, nChan, nTP, dimPosEmb]
        # Squeeze:
        y = x.transpose(1, 2) # [bs, nTP, nChan, dimPosEmb]
        y = self.squeezeBlock(y) # [bs, nTP, 1, 1]
        y = y.view(bs, nTP) # [bs, nTP]
        # Excitation:
        y = self.excitationBlock(y) # [bs, nTP]
        y = y.view(bs, nTP, 1, 1) # [bs, nTP, 1, 1]
        y = y.transpose(1, 2) # [bs, 1, nTP, 1]
        return x * y.expand_as(x) # [bs, nChan, nTP, dimPosEmb]


class ConvBlock(nn.Module):
    """
    """

    def __init__(self, batchnorm_dim, conv_in_chan=1, conv_out_chan=1, conv_kernel_shape=(1,3), conv_stride=(1,1), conv_padding=(0,1), activation='gelu', regularization=0):
        super().__init__()
        self.bn_dim = batchnorm_dim
        self.conv = nn.Conv2d(conv_in_chan, conv_out_chan, conv_kernel_shape, stride=conv_stride, padding=conv_padding)
        if regularization > 0.0:
            self.reg = nn.Dropout(regularization)
        elif regularization == -1.0:
            self.reg = nn.BatchNorm2d(self.bn_dim) # where self.bn_dim = conv_out_chan
            # self.reg = nn.BatchNorm1d(self.bn_dim) # TODO: Reconsider the normalization dimensionality
        else:
            self.reg = nn.Identity() # We use this to avoid if-else in forward pass

        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'mish':
            self.act = nn.Mish()
        else:
            raise ValueError('Unknown activation function type: %s'%activation)
                    

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.reg(x)
        return x


class MixerBlock(nn.Module):
    """
    """

    def __init__(self, 
                 channels_conv_dim, 
                 nTP, 
                 dimPosEmb, 
                 mode_conv="once",
                 activation='gelu', 
                 conv_kernel_shape=(1,3),
                 conv_stride=(1,1),
                 conv_padding=(0,1),
                 regularization=0,
                 use_se=True, 
                 r_se=4, 
                 use_max_pooling=False):
        super().__init__()

        self.channels_conv_dim = channels_conv_dim
        self.nTP = nTP # number of time points
        self.dimPosEmb = dimPosEmb  # dimension of the pose embedding
        self.mode_conv = mode_conv
        # self.conv_kernel_shape = conv_kernel_shape
        # self.conv_kernel_shape_transposed = (self.conv_kernel_shape[1], self.conv_kernel_shape[0])
        # self.conv_stride = conv_stride
        # self.conv_padding = conv_padding
        # self.conv_padding_transposed = (self.conv_padding[1], self.conv_padding[0])
        self.conv1 = ConvBlock(batchnorm_dim=self.channels_conv_dim, 
                               conv_in_chan=self.channels_conv_dim, 
                               conv_out_chan=self.channels_conv_dim, 
                               conv_kernel_shape=conv_kernel_shape, 
                               conv_stride=conv_stride, 
                               conv_padding=conv_padding,
                               activation=activation,
                               regularization=regularization)
        if use_se:
            self.se = MultiChanSELayer(self.nTP, r=r_se, use_max_pooling=use_max_pooling)
        else:
            self.se = nn.Identity()
        self.LN1 = nn.LayerNorm(self.dimPosEmb)
        
        if mode_conv == "twice":
            self.conv2 = ConvBlock(batchnorm_dim=self.channels_conv_dim,
                                conv_in_chan=self.channels_conv_dim,
                                conv_out_chan=self.channels_conv_dim,
                                conv_kernel_shape=(conv_kernel_shape[1], conv_kernel_shape[0]),
                                conv_stride=conv_stride,
                                conv_padding=(conv_padding[1], conv_padding[0]),
                                activation=activation,
                                regularization=regularization)
            self.se2 = self.se
            self.LN2 = nn.LayerNorm(self.dimPosEmb)
        elif mode_conv == "once":
            self.conv2 = nn.Identity()
            self.se2 = nn.Identity()
            self.LN2 = nn.Identity()
        else:
            raise ValueError("mode_conv %s"%mode_conv +" must be one of 'once' or 'twice'")


    def forward(self, x):
        """
        Forward pass of the MixerBlock
        Args:
        """
        # shape x [bs, nChan, nTP, dimPosEmb]
        y = self.LN1(x)

        # this should be the Spatial-Mix part according to the paper. 
        y = self.conv1(y)
        
        y = self.se(y)
        x = x + y

        # this should be the Temporal-Mix part according to the paper. 
        y = self.LN2(x)
        y = self.conv2(y) 
          
        y = self.se(y)
            
        return x + y


class ConvMixer(nn.Module):
    """

    """


    def __init__(self, 
                 num_blocks, 
                 nTP,
                 dimPosEmb,
                 num_out_classes, 
                 nChan=1,
                 conv_kernel_shape=(1,3),
                 conv_stride=(1,1),
                 conv_padding=(0,1),
                 outTP=15,
                 mode_conv="twice",
                 activation='gelu',
                 regularization=0, 
                 numJoints=51, 
                 use_se=False,
                 r_se=4, 
                 use_max_pooling=False):
        
        super().__init__()
        self.numJoints = numJoints # 51 for h36m, 66 for 3dpw
        self.num_out_classes = num_out_classes # 48 for h36m, 66 for 3dpw
        self.dimPosEmb = dimPosEmb
        self.num_blocks = num_blocks 
        self.nTP = nTP
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=self.dimPosEmb, kernel_size=(1, self.numJoints), stride=1)
        self.nChan = nChan
        self.channelUpscaling = nn.Linear(1, self.nChan)
        self.activation = activation
        
        self.Mixer_Block = nn.ModuleList(MixerBlock(channels_conv_dim = self.nChan, 
                                                    nTP=self.nTP, 
                                                    dimPosEmb=self.dimPosEmb, 
                                                    mode_conv=mode_conv,
                                                    activation=activation, 
                                                    conv_kernel_shape=conv_kernel_shape,
                                                    conv_stride=conv_stride,
                                                    conv_padding=conv_padding,
                                                    regularization=regularization,
                                                    use_se=use_se, 
                                                    r_se=r_se, 
                                                    use_max_pooling=use_max_pooling) 
                                                        for _ in range(num_blocks))
            
        
        self.LN = nn.LayerNorm(self.dimPosEmb)
        
        self.outTP = outTP
        self.project_channels = nn.Conv2d(self.nChan, 1, kernel_size=(1,1), stride=1)
        self.conv_out = nn.Conv1d(in_channels=self.nTP, out_channels=self.outTP, kernel_size=1, stride=1)
        self.fc_out = nn.Linear(self.dimPosEmb, self.num_out_classes)
        

    def forward(self, x):
        """
        """
        # NOTE: for the angle loss, 48 dimensions are used
        # for mpjpe loss, 66 dimensions are used
        # input x shape [batch_size, nTP, num_joints = 66 or 48]

        x = x.unsqueeze(1) # [bs, 1, nTP, numJoints]
        y = self.conv_in(x) # [bs, dimPosEmb, nTP, 1]

        y = self.channelUpscaling(y) # [bs, dimPosEmb, nTP, nChan]
        y = y.transpose(1, 3) # [bs, nChan, nTP, dimPosEmb]

        for mb in self.Mixer_Block:
            y = mb(y)  # [bs, nChan, nTP, dimPosEmb]
        y = self.LN(y) # [bs, nChan, nTP, dimPosEmb]

        y = self.project_channels(y).squeeze(1) # [bs, nTP, dimPosEmb]
        y = self.conv_out(y) # [bs, outTP, dimPosEmb]
        y = nn.GELU()(y) # [bs, outTP, dimPosEmb]
        out = self.fc_out(y) # [bs, outTP, num_out_classes]

        return out # [bs, outTP, num_out_classes]


def testOneForwardPass():
    num_blocks = 4
    nTP = 10
    dimPosEmb = 50
    conv_kernel_shape = (1,3)
    conv_stride = (1,1)
    conv_padding = (0,1)
    outTP = 15
    mode_conv = "twice"
    activation = 'gelu'
    regularization = 0
    numJoints = 66
    use_se = True
    r_se = 4
    use_max_pooling = False
    nChan=2

    model = ConvMixer(num_blocks=num_blocks,
                        nTP=nTP,
                        dimPosEmb=dimPosEmb,
                        num_out_classes=outTP,
                        conv_kernel_shape=conv_kernel_shape,
                        conv_stride=conv_stride,
                        conv_padding=conv_padding,
                        outTP=outTP,
                        mode_conv=mode_conv,
                        activation=activation,
                        regularization=regularization,
                        numJoints=numJoints,
                        use_se=use_se,
                        r_se=r_se,
                        use_max_pooling=use_max_pooling,
                        nChan=nChan)
    x = torch.randn(32, nTP, numJoints)
    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    testOneForwardPass()