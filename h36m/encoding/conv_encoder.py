import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, dimPosIn, in_nTP, dimPosEmb, conv_nChan):

        self.dimPosIn = dimPosIn # 51 for h36m, 66 for 3dpw
        self.in_nTP = in_nTP

        self.dimPosEmb = dimPosEmb
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=self.dimPosEmb, kernel_size=(1, self.dimPosIn), stride=1)
        self.conv_nChan = conv_nChan
        self.channelUpscaling = nn.Linear(1, self.conv_nChan)


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

        return y
