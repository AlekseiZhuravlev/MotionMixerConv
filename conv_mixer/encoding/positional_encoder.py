import torch
import torch.nn as nn

class PoseEncoder(nn.Module):
    def __init__(self, dimPosIn, in_nTP, dimPosEmb, conv_nChan, n_harmonic_functions, omega0):
        """
        Encoding of the joint position coordinates, with an optional harmonic embedding.

        Harmonic embedding:
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]
        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.

        Args:
            dimPosIn (int):
                Pose dimension of the input tensor
            in_nTP (int):
                Number of time points in the input tensor
            dimPosEmb (int):
                Pose dimension of the output tensor
            conv_nChan (int):
                Number of channels in the output tensor
            n_harmonic_functions (int):
                Number of harmonic functions to use
                If <= 0, do not use harmonic embedding
            omega0 (float), (default=0.1):
                Base frequency of the harmonic functions
        """
        super().__init__()

        self.n_harmonic_functions = n_harmonic_functions

        # dimPosIn == 48 for h36m, 66 for 3dpw

        if n_harmonic_functions <= 0:
            # do not use harmonic embedding
            dimHarmonic = dimPosIn
        else:
            self.register_buffer(
                'frequencies',
                omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
            )
            dimHarmonic = n_harmonic_functions * dimPosIn * 2

        # self.conv_in = nn.Conv2d(in_channels=1, out_channels=dimPosEmb, kernel_size=(1, dimHarmonic), stride=1)

        self.embed_mlp = nn.Linear(dimHarmonic, dimPosEmb)
        self.channelUpscaling = nn.Linear(1, conv_nChan)


    def forward(self, x):
        """
        Args:
            x: tensor of shape [bs, in_nTP, dimPosIn]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [bs, conv_nChan, in_nTP, dimPosEmb = n_harmonic_functions * dim * 2]
        """
        # NOTE: for the angle loss, 48 dimensions are used
        # for mpjpe loss, 66 dimensions are used
        # input x shape [batch_size, in_nTP, num_joints = 66 or 48]


        x = x.unsqueeze(1) # [bs, 1, in_nTP, dimPosIn]

        if self.n_harmonic_functions <= 0:
            # do not use harmonic embedding
            embed = x
        else:
            # [bs, 1, in_nTP, n_harmonic_functions * dimPosIn]
            embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)

            # [bs, 1, in_nTP, self.dimHarmonic = n_harmonic_functions * dimPosIn * 2]
            embed = torch.cat((embed.sin(), embed.cos()), dim=-1)

        y = self.embed_mlp(embed) # [bs, 1, in_nTP, dimPosEmb]
        y = y.transpose(1, 3) # [bs, dimPosEmb, in_nTP, 1]

        y = self.channelUpscaling(y) # [bs, dimPosEmb, in_nTP, conv_nChan]
        y = y.transpose(1, 3) # [bs, conv_nChan, in_nTP, dimPosEmb]

        return y


if __name__ == '__main__':
    pos_encoder = PoseEncoder(dimPosIn=66, in_nTP=32, dimPosEmb=50, conv_nChan=3, n_harmonic_functions=-1)
    # print n of parameters
    print (sum(p.numel() for p in pos_encoder.parameters() if p.requires_grad))

    x = torch.rand(1, 32, 66)
    print (x.shape)
    y = pos_encoder(x)
    print (y.shape)


