import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, dimPosIn, in_nTP, dimPosEmb, conv_nChan, n_harmonic_functions, omega0=0.1):
        """
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
        """
        super().__init__()

        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

        # self.dimPosIn = dimPosIn # 51 for h36m, 66 for 3dpw
        self.dimHarmonic = n_harmonic_functions * dimPosIn * 2
        self.in_nTP = in_nTP

        self.dimPosEmb = dimPosEmb
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=self.dimPosEmb, kernel_size=(1, self.dimHarmonic), stride=1)
        self.conv_nChan = conv_nChan
        self.channelUpscaling = nn.Linear(1, self.conv_nChan)


    def forward(self, x):
        """
        Args:
            x: tensor of shape [bs, 1, in_nTP, dimPosIn]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [bs, conv_nChan, in_nTP, dimPosEmb = n_harmonic_functions * dim * 2]
        """
        # NOTE: for the angle loss, 48 dimensions are used
        # for mpjpe loss, 66 dimensions are used
        # input x shape [batch_size, in_nTP, num_joints = 66 or 48]


        x = x.unsqueeze(1) # [bs, 1, in_nTP, dimPosIn]

        print (x.shape)

        # [bs, 1, in_nTP, n_harmonic_functions * dim]
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)

        print (embed.shape)

        # [bs, 1, in_nTP, self.dimHarmonic = n_harmonic_functions * dim * 2]
        embed = torch.cat((embed.sin(), embed.cos()), dim=-1)

        print (embed.shape)

        y = self.conv_in(embed) # [bs, dimPosEmb, in_nTP, 1]

        print (y.shape)

        y = self.channelUpscaling(y) # [bs, dimPosEmb, in_nTP, conv_nChan]

        print (y.shape)
        y = y.transpose(1, 3) # [bs, conv_nChan, in_nTP, dimPosEmb]
        print (y.shape)

        return y


if __name__ == '__main__':
    pos_encoder = ConvEncoder(dimPosIn=66, in_nTP=32, dimPosEmb=50, conv_nChan=3, n_harmonic_functions=4)

    x = torch.rand(1, 32, 66)
    print (x.shape)
    y = pos_encoder(x)
    print (y.shape)

    # TODO this could be implemented in a single class

