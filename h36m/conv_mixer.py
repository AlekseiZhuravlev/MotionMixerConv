import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional


class MultiChanSELayer(nn.Module):
    """
        Squeeze-and-Excitation layer for Tensors of Shape [bs, conv_nChan, in_nTP, dimPosEmb].
        If conv_nChan = 1, this is equivalent to the original SE-Layer. 
        Valentin: I am not 100% satisfied with the MotionMixer-Paper at this point: 
        SE-Architectures were originally designed to weight a bunch of channels 
        and not a bunch of time points.
        TODO: In later experiments, we could set our number of channels (conv_nChan) to
            a value > 1 and apply the SE-Layer as planned by the original SE-paper 
            authors: I.e.  [bs, conv_nChan, in_nTP, dimPosEmb] -(squeeze)-> [bs, conv_nChan] 
            -> [bs, conv_nChan // r] -> [bs, conv_nChan] -> [bs, conv_nChan, 1, 1] -> [bs, conv_nChan, in_nTP, dimPosEmb]
    """
    def __init__(self, in_nTP, r=4, use_max_pooling=False):
        super().__init__()
        self.squeezeBlock = nn.AdaptiveAvgPool2d((1,1)) if not use_max_pooling else nn.AdaptiveMaxPool2d((1,1))
        self.excitationBlock = nn.Sequential(
            nn.Linear(in_nTP, in_nTP // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_nTP // r, in_nTP, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x): # Input: [bs, conv_nChan, in_nTP, dimPosEmb]
        bs, _, in_nTP, _ = x.shape # [bs, conv_nChan, in_nTP, dimPosEmb]
        # Squeeze:
        y = x.transpose(1, 2) # [bs, in_nTP, conv_nChan, dimPosEmb]
        y = self.squeezeBlock(y) # [bs, in_nTP, 1, 1]
        y = y.view(bs, in_nTP) # [bs, in_nTP]
        # Excitation:
        y = self.excitationBlock(y) # [bs, in_nTP]
        y = y.view(bs, in_nTP, 1, 1) # [bs, in_nTP, 1, 1]
        y = y.transpose(1, 2) # [bs, 1, in_nTP, 1]
        return x * y.expand_as(x) # [bs, conv_nChan, in_nTP, dimPosEmb]


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
                 dimPosEmb:int,
                 in_nTP:int, 
                 conv_nChan:int,  
                 conv1_kernel_shape:Tuple[int, int]=(1,3),
                 conv1_stride:Tuple[int, int]=(1,1),
                 conv1_padding:Union[Tuple[int, int], None]=None, 
                 mode_conv:str="twice",
                 conv2_kernel_shape:Union[Tuple[int, int], None]=None, 
                 conv2_stride:Union[Tuple[int, int], None]=None,
                 conv2_padding:Union[Tuple[int, int], None]=None,
                 activation:str='gelu',
                 regularization:float=0,
                 use_se:bool=True, 
                 r_se:int=4, 
                 use_max_pooling:bool=False):
        super().__init__()

        self.conv_nChan = conv_nChan
        self.in_nTP = in_nTP # number of time points
        self.dimPosEmb = dimPosEmb  # dimension of the pose embedding
        self.mode_conv = mode_conv
        # self.conv_kernel_shape = conv_kernel_shape
        # self.conv_kernel_shape_transposed = (self.conv_kernel_shape[1], self.conv_kernel_shape[0])
        # self.conv_stride = conv_stride
        # self.conv_padding = conv_padding
        # self.conv_padding_transposed = (self.conv_padding[1], self.conv_padding[0])
        if conv1_padding is None:
            # Auto-padding
            conv1_padding = (conv1_kernel_shape[0]//2, conv1_kernel_shape[1]//2)
        self.conv1 = ConvBlock(batchnorm_dim=self.conv_nChan, 
                               conv_in_chan=self.conv_nChan, 
                               conv_out_chan=self.conv_nChan, 
                               conv_kernel_shape=conv1_kernel_shape, 
                               conv_stride=conv1_stride, 
                               conv_padding=conv1_padding,
                               activation=activation,
                               regularization=regularization)
        if use_se:
            self.se = MultiChanSELayer(self.in_nTP, r=r_se, use_max_pooling=use_max_pooling)
        else:
            self.se = nn.Identity()
        self.LN1 = nn.LayerNorm(self.dimPosEmb)
        
        if mode_conv == "twice":
            if conv2_kernel_shape is None:
                conv2_kernel_shape = conv1_kernel_shape[::-1]
            if conv2_stride is None:
                conv2_stride = conv1_stride[::-1]
            if conv2_padding is None:
                # Auto-padding
                conv2_padding = (conv2_kernel_shape[0]//2, conv2_kernel_shape[1]//2)
            self.conv2 = ConvBlock(batchnorm_dim=self.conv_nChan,
                                conv_in_chan=self.conv_nChan,
                                conv_out_chan=self.conv_nChan,
                                conv_kernel_shape=conv2_kernel_shape,
                                conv_stride=conv2_stride,
                                conv_padding=conv2_padding,
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


    def forward(self, x:torch.Tensor):
        """
        Forward pass of the MixerBlock
        Args:
        """
        # shape x [bs, conv_nChan, in_nTP, dimPosEmb]
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
                 num_blocks:int, 
                 dimPosIn:int,
                 dimPosEmb:int,
                 dimPosOut:int, 
                 in_nTP:int,
                 out_nTP:int,
                 conv_nChan:int=1,
                 conv1_kernel_shape:Tuple[int, int]=(1,3),
                 conv1_stride:Tuple[int, int]=(1,1),
                 conv1_padding:Union[Tuple[int, int], None]=(0,1), 
                 mode_conv:str="twice",
                 conv2_kernel_shape:Union[Tuple[int, int], None]=None, 
                 conv2_stride:Union[Tuple[int, int], None]=None, 
                 conv2_padding:Union[Tuple[int, int], None]=None, 
                 activation:str='gelu',
                 regularization:float=0,  
                 use_se:bool=False,
                 r_se:int=4, 
                 use_max_pooling:bool=False):
        
        super().__init__()
        self.dimPosIn = dimPosIn # 51 for h36m, 66 for 3dpw
        self.dimPosOut = dimPosOut # 48 for h36m, 66 for 3dpw
        self.dimPosEmb = dimPosEmb
        self.num_blocks = num_blocks 
        self.in_nTP = in_nTP
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=self.dimPosEmb, kernel_size=(1, self.dimPosIn), stride=1)
        self.conv_nChan = conv_nChan
        self.channelUpscaling = nn.Linear(1, self.conv_nChan)
        self.activation = activation
        
        self.Mixer_Block = nn.ModuleList(MixerBlock(dimPosEmb=self.dimPosEmb,
                                                    in_nTP=self.in_nTP,
                                                    conv_nChan = self.conv_nChan, 
                                                    conv1_kernel_shape=conv1_kernel_shape,
                                                    conv1_stride=conv1_stride,
                                                    conv1_padding=conv1_padding,
                                                    mode_conv=mode_conv,
                                                    conv2_kernel_shape=conv2_kernel_shape,
                                                    conv2_stride=conv2_stride,
                                                    conv2_padding=conv2_padding,
                                                    activation=activation,
                                                    regularization=regularization,
                                                    use_se=use_se, 
                                                    r_se=r_se, 
                                                    use_max_pooling=use_max_pooling) 
                                                        for _ in range(num_blocks))
        
        self.LN = nn.LayerNorm(self.dimPosEmb)
        
        self.out_nTP = out_nTP
        self.project_channels = nn.Conv2d(self.conv_nChan, 1, kernel_size=(1,1), stride=1)
        self.conv_out = nn.Conv1d(in_channels=self.in_nTP, out_channels=self.out_nTP, kernel_size=1, stride=1)
        self.fc_out = nn.Linear(self.dimPosEmb, self.dimPosOut)
        

    def forward(self, x:torch.Tensor):
        """
        """
        # NOTE: for the angle loss, 48 dimensions are used
        # for mpjpe loss, 66 dimensions are used
        # input x shape [batch_size, in_nTP, num_joints = 66 or 48]

        ##### Encoding #####
        x = x.unsqueeze(1) # [bs, 1, in_nTP, dimPosIn]
        y = self.conv_in(x) # [bs, dimPosEmb, in_nTP, 1]

        y = self.channelUpscaling(y) # [bs, dimPosEmb, in_nTP, conv_nChan]
        y = y.transpose(1, 3) # [bs, conv_nChan, in_nTP, dimPosEmb]

        ##### Mixer Blocks #####
        for mb in self.Mixer_Block:
            y = mb(y)  # [bs, conv_nChan, in_nTP, dimPosEmb]
        y = self.LN(y) # [bs, conv_nChan, in_nTP, dimPosEmb]

        ##### Decoding #####
        y = self.project_channels(y).squeeze(1) # [bs, in_nTP, dimPosEmb]
        y = self.conv_out(y) # [bs, out_nTP, dimPosEmb]
        y = nn.GELU()(y) # [bs, out_nTP, dimPosEmb]
        out = self.fc_out(y) # [bs, out_nTP, dimPosOut]

        return out # [bs, out_nTP, dimPosOut]


def testOneForwardPass():
    num_blocks = 4
    in_nTP = 10
    dimPosEmb = 50
    conv_kernel_shape = (1,3)
    conv_stride = (1,1)
    conv_padding = (0,1)
    out_nTP = 15
    mode_conv = "twice"
    activation = 'gelu'
    regularization = 0
    numJoints = 66
    use_se = True
    r_se = 4
    use_max_pooling = False
    conv_nChan=2

    model = ConvMixer(num_blocks=num_blocks,
                        dimPosIn=numJoints,
                        dimPosEmb=dimPosEmb,
                        dimPosOut=numJoints,
                        in_nTP=in_nTP,
                        out_nTP=out_nTP,
                        conv_nChan=conv_nChan,
                        conv1_kernel_shape=conv_kernel_shape,
                        conv1_stride=conv_stride,
                        conv1_padding=conv_padding,
                        mode_conv=mode_conv,
                        activation=activation,
                        regularization=regularization,
                        use_se=use_se,
                        r_se=r_se,
                        use_max_pooling=use_max_pooling)
    x = torch.randn(32, in_nTP, numJoints)
    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    testOneForwardPass()