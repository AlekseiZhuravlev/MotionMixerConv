import torch.nn as nn
import torch
import torch.nn.functional as F


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation layer
    First (squeeze), performs avg/max pooling to reduce the number of last dimensions to 1,
    e.g. (N,C,L_in) -> (N, C, 1)
    Then (excitation), applies two FC layers, c -> c // r and c // r -> c,
     with ReLU and Sigmoid activations

    c (int): number of input channels for excitation block
    r (int): reduction ratio for excitation block. Default: 4
    use_max_pooling (bool): if True, uses nn.AdaptiveMaxPool1d instead of nn.AdaptiveAvgPool1d. Default: False

    """
    def __init__(self, c, r=4, use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        bs, s, h = x.shape
        y = self.squeeze(x).view(bs, s)
        y = self.excitation(y).view(bs, s, 1)
        return x * y.expand_as(x)


def mish(x):
    """
    Mish activation function
    """
    return (x*torch.tanh(F.softplus(x)))


class MlpBlock(nn.Module):
    """
    A 2 layer MLP with activation functions and regularization
    First, applies a linear layer, then an activation function (gelu or mish),
    then a regularization layer (dropout or batchnorm1d)
    Second, another linear layer, an activation function, and a regularization layer

    Args:
        mlp_hidden_dim (int): number of hidden units in the first linear layer
        mlp_input_dim (int): number of input units in the first linear layer
        mlp_bn_dim (int): number of units in the batchnorm1d layer
        activation (str): activation function to use, either 'gelu' or 'mish'
        regularization  (int): regularization to use, dropout if > 0.0, batchnorm1d if -1.0, None otherwise
        initialization (str): not used
    """

    def __init__(self, mlp_hidden_dim, mlp_input_dim, mlp_bn_dim, activation='gelu', regularization=0, initialization='none'):
        super().__init__()
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_input_dim = mlp_input_dim
        self.mlp_bn_dim = mlp_bn_dim
        #self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_input_dim)
        self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_hidden_dim)
        self.fc2 = nn.Linear(self.mlp_hidden_dim, self.mlp_input_dim)
        if regularization > 0.0:
            self.reg1 = nn.Dropout(regularization)
            self.reg2 = nn.Dropout(regularization)
        elif regularization == -1.0:
            self.reg1 = nn.BatchNorm1d(self.mlp_bn_dim)
            self.reg2 = nn.BatchNorm1d(self.mlp_bn_dim)
        else:
            self.reg1 = None
            self.reg2 = None

        if activation == 'gelu':
            self.act1 = nn.GELU()
        elif activation == 'mish':
            self.act1 = mish #nn.Mish()
        else:
            raise ValueError('Unknown activation function type: %s'%activation)
            
                    

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        if self.reg1 is not None:
            x = self.reg1(x)
        x = self.fc2(x)
        if self.reg2 is not None:
            x = self.reg2(x)
            
        return x



class MixerBlock(nn.Module):
    """
    MixerBlock as described in the paper, with spatial and temporal mixing.
    First, applies a layer normalization, transpose dim 1-2,
    then a spatial mixing MLP block (with optional squeeze-and-excitation), and transpose dim 1-2 again.
    Output is added to the input (residual connection).
    Second, applies another layer normalization,
    then a temporal mixing MLP block (with optional squeeze-and-excitation) and a residual connection.

    Args:
        tokens_mlp_dim (int): number of hidden units in the first linear layer of the spatial mixing MLP
        channels_mlp_dim (int): number of hidden units in the first linear layer of the temporal mixing MLP
        seq_len (int): length of the input sequence
        hidden_dim (int): number of output channels of the LayerNorm
        activation (str): activation function to use, either 'gelu' or 'mish'
        regularization  (int): regularization to use in MlpBlocks, dropout if > 0.0, batchnorm1d if -1.0, None otherwise
        initialization (str): initialization to be used in MlpBlocks, but not used
        r_se (int): reduction ratio for squeeze-and-excitation blocks. Default: 4
        use_max_pooling (bool): if True, uses nn.AdaptiveMaxPool1d instead of nn.AdaptiveAvgPool1d. Default: False
        use_se (bool): if True, uses squeeze-and-excitation blocks. Default: True
    """

    def __init__(self, tokens_mlp_dim, channels_mlp_dim, seq_len, hidden_dim, activation='gelu', regularization=0,
                 initialization='none', r_se=4, use_max_pooling=False, use_se=True):
        super().__init__()
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim  # out channels of the conv
        self.mlp_block_token_mixing = MlpBlock(self.tokens_mlp_dim, self.seq_len, self.hidden_dim, activation=activation, regularization=regularization, initialization=initialization)
        self.mlp_block_channel_mixing = MlpBlock(self.channels_mlp_dim, self.hidden_dim, self.seq_len, activation=activation, regularization=regularization, initialization=initialization)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(self.seq_len, r=r_se, use_max_pooling=use_max_pooling)

        self.LN1 = nn.LayerNorm(self.hidden_dim)
        self.LN2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        """
        Forward pass of the MixerBlock
        Args:
            x (torch.Tensor): input tensor of shape [bs, patches/time_steps, channels]
             e.g. [256, 8, 512]
        """
        # shape x [256, 8, 512] [bs, patches/time_steps, channels
        y = self.LN1(x)

        # this should be the Spatial-Mix part according to the paper. I may be wrong
        y = y.transpose(1, 2)
        y = self.mlp_block_token_mixing(y)
        y = y.transpose(1, 2)
        
        if self.use_se:
            y = self.se(y)
        x = x + y

        # this should be the Temporal-Mix part according to the paper. I may be wrong
        y = self.LN2(x)
        y = self.mlp_block_channel_mixing(y)  
          
        if self.use_se:
            y = self.se(y)
            
        return x + y

class MixerBlock_Channel(nn.Module):
    """
    Same as MixerBlock, but only uses temporal mixing.
    """

    def __init__(self, channels_mlp_dim, seq_len, hidden_dim, activation='gelu', regularization=0,
                 initialization='none', r_se=4, use_max_pooling=False, use_se=True):
        super().__init__()
        self.channels_mlp_dim = channels_mlp_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim  # out channels of the conv
        self.mlp_block_channel_mixing = MlpBlock(self.channels_mlp_dim, self.hidden_dim, self.seq_len, activation=activation, regularization=regularization, initialization=initialization)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(self.seq_len, r=r_se, use_max_pooling=use_max_pooling)

        
        self.LN2 = nn.LayerNorm(self.hidden_dim)
        
        #self.act1 = nn.GELU()

    def forward(self, x):
        # shape x [256, 8, 512] [bs, patches/time_steps, channels]
        y = x
        
        if self.use_se:
            y = self.se(y)
        x = x + y
        y = self.LN2(x)
        y = self.mlp_block_channel_mixing(y)            
        if self.use_se:
            y = self.se(y)
            
        return x + y
    
    
    
class MixerBlock_Token(nn.Module):
    """
    Same as MixerBlock, but only uses spatial mixing.
    """

    def __init__(self, tokens_mlp_dim,  seq_len, hidden_dim, activation='gelu', regularization=0,
                 initialization='none', r_se=4, use_max_pooling=False, use_se=True):
        super().__init__()
        self.tokens_mlp_dim = tokens_mlp_dim

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim  # out channels of the conv
        self.mlp_block_token_mixing = MlpBlock(self.tokens_mlp_dim, self.seq_len, self.hidden_dim, activation=activation, regularization=regularization, initialization=initialization)
    
        self.use_se = use_se
        
        if self.use_se:
            self.se = SELayer(self.seq_len, r=r_se, use_max_pooling=use_max_pooling)

        self.LN1 = nn.LayerNorm(self.hidden_dim)
        

    def forward(self, x):
        # shape x [256, 8, 512] [bs, patches/time_steps, channels]
        y = self.LN1(x)
        y = y.transpose(1, 2)
        y = self.mlp_block_token_mixing(y)
        y = y.transpose(1, 2)
        
        if self.use_se:
            y = self.se(y)
        x = x + y

        return x + y


class MlpMixer(nn.Module):
    """
    Main model of the paper.
    First, applies a conv1d to the input, then sequentially applies all MixerBlocks.
    Then, applies LayerNorm, another conv1d, and a fully connected layer to get the output.

    Args:
        num_classes (int): dimension of the output of last fc layer, should be pose_dim
        num_blocks (int): number of MixerBlocks
        hidden_dim (int): passed to MixerBlock
        tokens_mlp_dim (int): passed to MixerBlock
        channels_mlp_dim (int): passed to MixerBlock
    """


    def __init__(self, num_classes, num_blocks, hidden_dim, tokens_mlp_dim, 
                 channels_mlp_dim, seq_len,pred_len, activation='gelu', 
                 mlp_block_type='normal',regularization=0, input_size=51, 
                 initialization='none', r_se=4, use_max_pooling=False, 
                 use_se=False):
        
        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.input_size = input_size #varyies with the number of joints
        self.conv = nn.Conv1d(1, self.hidden_dim, (1, self.input_size), stride=1)
        self.activation = activation

        self.channel_only = False # False #True
        self.token_only = False #False #True
        
        if self.channel_only:
            
            self.Mixer_Block = nn.ModuleList (MixerBlock_Channel(self.channels_mlp_dim,self.seq_len, self.hidden_dim,
                                              activation=self.activation, regularization=regularization, initialization=initialization,
                                              r_se=r_se, use_max_pooling=use_max_pooling, use_se=use_se)
                                              for _ in range(num_blocks))
                
                
        if self.token_only:
            
            self.Mixer_Block = nn.ModuleList(MixerBlock_Token(self.tokens_mlp_dim, self.seq_len, self.hidden_dim,
                                                      activation=self.activation, regularization=regularization, initialization=initialization,
                                                      r_se=r_se, use_max_pooling=use_max_pooling, use_se=use_se)
                                                      for _ in range(num_blocks))
                                        
        else:
            
            self.Mixer_Block = nn.ModuleList(MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim, 
                                                        self.seq_len, self.hidden_dim, activation=self.activation, 
                                                        regularization=regularization, initialization=initialization,
                                                        r_se=r_se, use_max_pooling=use_max_pooling, use_se=use_se) 
                                                        for _ in range(num_blocks))
            
        
        self.LN = nn.LayerNorm(self.hidden_dim)
        
        self.fc_out = nn.Linear(self.hidden_dim, self.num_classes)
        
        self.pred_len = pred_len
        self.conv_out = nn.Conv1d(self.seq_len, self.pred_len, 1, stride=1)
        

    def forward(self, x):
        """

        x shape after unsqueeze torch.Size([50, 1, 10, 66])
        y shape torch.Size([50, 50, 10, 1])
        y shape after transpose torch.Size([50, 10, 50])
        y shape after mixer block torch.Size([50, 10, 50])
        y shape after mixer block torch.Size([50, 10, 50])
        y shape after mixer block torch.Size([50, 10, 50])
        y shape after mixer block torch.Size([50, 10, 50])
        y shape after LN torch.Size([50, 10, 50])
        y shape after conv_out torch.Size([50, 25, 50])
        out shape torch.Size([50, 25, 66])
        """
        # NOTE: for the angle loss, 48 dimensions are used
        # for mpjpe loss, 66 dimensions are used
        # x shape [50, 10, 66]

        print("x shape", x.shape)
        exit(0)

        # pose embedding, input_size -> hidden_dim (66 -> 50)
        x = x.unsqueeze(1) # [50, 1, 10, 66]
        y = self.conv(x) # [50, 50, 10, 1]
        y = y.squeeze(dim=3).transpose(1, 2) # [50, 10, 50]

        # [256, 8, 512] [bs, patches/time_steps, channels] this is for AMASS
        for mb in self.Mixer_Block:
            y = mb(y) # [50, 10, 50]
        y = self.LN(y) #[50, 10, 50]

        y = self.conv_out(y) # [50, 25, 50]
        out = self.fc_out(y) # [50, 25, 66]

        return out
