# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional,Tuple
import os
base_path=os.environ.get("project_path")
import math
from torchvision import models as vision_models
import abc
from typing import List
from collections import OrderedDict
import textwrap
import torch.nn.functional as F
from torchvision import transforms
from typing import Union
from math import ceil,sqrt
from torch.nn.parallel import DistributedDataParallel as DDP

# %%
#############
#Base Module#
#############
class Module(nn.Module):
    """
    Base class for networks. The only difference from torch.nn.Module is that it
    requires implementing @output_shape.
    """
    def __init__(self):
        super().__init__()
    @abc.abstractmethod
    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError
    
    def forward(self,inputs):
        x = self.layers(inputs)
        if list(self.output_shape(list(inputs.shape))) != list(x.shape):
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape))), str(list(x.shape)))
            )
        return x
    
class Sequential(nn.Sequential):
    """
    Compose multiple Modules together (defined above).
    """
    def __init__(self, *args):
        """
        Args:
            has_output_shape (bool, optional): indicates whether output_shape can be called on the Sequential module.
                torch.nn modules do not have an output_shape, but Modules (defined above) do. Defaults to True.
        """
        super().__init__(*args)
    
    
    def output_shape(self, input_shape:List[torch.Size]=None)->List[torch.Size]:
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        out_shape = input_shape
        for module in self:
            out_shape = module.output_shape(out_shape)
        return out_shape



# %%
#############
# Activation#
#############
class ReLU(Module):
    def __init__(self,inplace:bool=False):
        super().__init__()
        self.layers=nn.ReLU(inplace=inplace)
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Tanh()
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sigmoid()
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class Leaky_RELU(Module):
    def __init__(self,negative_slope:float=0.01,inplace:bool=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.layers=nn.LeakyReLU(negative_slope=negative_slope,inplace=inplace)
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def __repr__(self):
        return f"{self.__class__.__name__}(negative_slope={self.negative_slope})"

class GELU(Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.GELU()
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def __repr__(self):
        return f"{self.__class__.__name__}()"

activation_map=dict(relu=ReLU, tanh=Tanh, sigmoid=Sigmoid, leaky_relu=Leaky_RELU,gelu=GELU)


# %%
#########
# Linear#
# #######   
class InvertLayer(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return -x
    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class Flatten(Module):
    def __init__(self,start_dim:int=1,end_dim:int=-1):
        super().__init__()
        self.start_dim=start_dim
        self.end_dim=end_dim
        self.layers=nn.Flatten(start_dim=start_dim,end_dim=end_dim)
    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        start_dim = self.start_dim if self.start_dim >= 0 else len(input_shape) + self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(input_shape) + self.end_dim
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= input_shape[i]
        return input_shape[:start_dim] + [flattened_size] + input_shape[end_dim + 1:]
    def __repr__(self):
        return f"{self.__class__.__name__}(start_dim={self.start_dim}, end_dim={self.end_dim})"
        
class Linear(Module):
    def __init__(self,input_dim:int,output_dim:int,bias:bool=False,spectral_norm:bool=False,**kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers=nn.Linear(in_features=input_dim,out_features=output_dim,bias=bias,**kwargs)
        if spectral_norm:
            self.layers = spectral_norm(self.layers)
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape[:-1]+[self.output_dim]
    # def forward(self, inputs):
    #     x=self.layers(inputs)
    #     if list(self.output_shape(list(inputs.shape))) != list(x.shape):
    #         raise ValueError('Size mismatch: expect size %s, but got size %s' % (
    #             str(self.output_shape(list(inputs.shape))), str(list(x.shape)))
    #         )
        # return x
    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim},output_dim={self.output_dim})"

class LayerNorm(Module):
    def __init__(self,normalized_shape:int):
        super().__init__()
        self.normalized_shape=normalized_shape
        self.layers=nn.LayerNorm(normalized_shape=normalized_shape)
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def __repr__(self):
        return f"{self.__class__.__name__}(normalized_shape={self.normalized_shape})"

class BatchNorm2d(Module):
    def __init__(self,num_feature:int):
        super().__init__()
        self.num_feature=num_feature
        self.layers=nn.BatchNorm2d(num_features=num_feature)
    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def __repr__(self):
        return f"{self.__class__.__name__}(num_feature={self.num_feature})"

class BatchNorm1d(Module):
    def __init__(self,num_feature:int):
        super().__init__()
        self.num_feature=num_feature
        self.layers=nn.BatchNorm1d(num_features=num_feature)
    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def __repr__(self):
        return f"{self.__class__.__name__}(num_feature={self.num_feature})"
    
class Dropout(Module):
    def __init__(self,p:int,**kwargs):
        super().__init__()
        self.p=p
        self.layers=nn.Dropout(p=self.p,**kwargs)
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
    
class Softmax(Module):
    def __init__(self,dim:int=-1):
        super().__init__()
        self.dim = dim 
        self.layers=nn.Softmax(dim=self.dim)
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def __repr__(self):
        return f"{self.__class__.__name__}(output_dim={self.dim})"
    
class MLP(Module):
    """
    Base class for simple Multi-Layer Perceptrons.
    """
    def __init__(
        self,
        input_dim:int,
        output_dim:int,
        layer_dims:List[int]=[],
        activation:str="relu",
        dropouts:List[float]=None,
        normalization:bool=False,
        output_activation:Optional[str]=None,
        **layer_func_kwargs
    ):
        """
        Args:
            input_dim (int): dimension of inputs

            output_dim (int): dimension of outputs

            layer_dims ([int]): sequence of integers for the hidden layers sizes

            layer_func: mapping per layer - defaults to Linear

            layer_func_kwargs (dict): kwargs for @layer_func

            activation (str): non-linearity per layer - defaults to ReLU

            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.

            normalization (bool): if True, apply layer normalization after each layer

            output_activation(Optional[str]): if provided, applies the provided non-linearity to the output layer
        """
        super().__init__()
        self.layers = []
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropouts = dropouts
        self.activation = activation
        self.output_activation = output_activation
        dim = input_dim
        if dropouts is not None:
            assert(len(dropouts) == len(layer_dims))
        for i, l in enumerate(layer_dims):
            self.layers.append(Linear(input_dim=dim, output_dim=l, **layer_func_kwargs))
            if normalization:
                self.layers.append(LayerNorm(l))
            self.layers.append(activation())
            if dropouts is not None and dropouts[i] > 0.:
                self.layers.append(Dropout(dropouts[i]))
            dim = l
        self.layers.append(Linear(dim, output_dim))
        if output_activation is not None:
            self.layers.append(output_activation())
        self.layers = nn.Sequential(*self.layers)

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [self._output_dim]

    def forward(self, inputs):
        """
        Forward pass.
        """
        return self._model(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = str(self.__class__.__name__)
        activation = None if self.activation is None else self.activation
        output_activation = None if self.output_activation is None else self.output_activation

        indent = ' ' * 4
        msg = "input_dim={}\noutput_dim={}\nlayer_dims={}\ndropout={}\nact={}\noutput_act={}".format(
            self.input_dim, self.output_dim, self.layer_dims,
            self.dropouts, activation, output_activation
        )
        msg = textwrap.indent(msg, indent)
        msg = header + '(\n' + msg + '\n)'
        return msg




# %%


# %%
######
#Conv#
######
class Conv2d(Module):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:Union[int,Tuple[int,int]],stride:Union[int,Tuple[int,int]]=1,padding:Union[int,str,Tuple[int,int]]=0,**kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.layers=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=self.padding,**kwargs)
    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        if isinstance(self.kernel_size,int):
            self.kernel_size=[self.kernel_size]*2
        if isinstance(self.stride,int):
            self.stride=[self.stride]*2
        if isinstance(self.padding,str):
            if self.padding == 'same':
                return [input_shape[0],self.out_channels]+input_shape[-2:]
            elif self.padding == 'valid':
                self.padding = [0,0]
        elif isinstance(self.padding,int):
            self.padding=[self.padding]*2
        assert input_shape[1]==self.in_channels
        im_h= (input_shape[2] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        im_w = (input_shape[3] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        out_shape=[input_shape[0],self.out_channels,im_h,im_w]
        return out_shape
    def __repr__(self):
        """Pretty print network."""
        return f"{self.__class__.__name__}(input_channel={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride},padding={self.padding})"

class R3MConv(Module):
    """
    Base class for ConvNets pretrained with R3M (https://arxiv.org/abs/2203.12601)
    """
    def __init__(
        self,
        input_channel=3,
        r3m_model_class='resnet18',
        freeze=True,
    ):
        """
        Using R3M pretrained observation encoder network proposed by https://arxiv.org/abs/2203.12601
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            r3m_model_class (str): select one of the r3m pretrained model "resnet18", "resnet34" or "resnet50"
            freeze (bool): if True, use a frozen R3M pretrained model.
        """
        super(R3MConv, self).__init__()

        try:
            from r3m import load_r3m
        except ImportError:
            print("WARNING: could not load r3m library! Please follow https://github.com/facebookresearch/r3m to install R3M")

        net = load_r3m(r3m_model_class)

        assert input_channel == 3 # R3M only support input image with channel size 3
        assert r3m_model_class in ["resnet18", "resnet34", "resnet50"] # make sure the selected r3m model do exist

        # cut the last fc layer
        self._input_channel = input_channel
        self._r3m_model_class = r3m_model_class
        self._freeze = freeze
        self._input_coord_conv = False
        self._pretrained = True

        preprocess = nn.Sequential(
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        self.layers = nn.Sequential(*([preprocess] + list(net.module.convnet.children())), has_output_shape = False)

        self.weight_sum = np.sum([param.cpu().data.numpy().sum() for param in self.nets.parameters()])
        if freeze:
            self.layers.train(False)
            for param in self.nets.parameters():
                param.requires_grad = False

        self.nets.eval()

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)

        if self._r3m_model_class == 'resnet50':
            out_dim = 2048
        else:
            out_dim = 512

        return [out_dim, 1, 1]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={}, pretrained={}, freeze={})'.format(self._input_channel, self._input_coord_conv, self._pretrained, self._freeze)


class MVPConv(Module):
    """
    Base class for ConvNets pretrained with MVP (https://arxiv.org/abs/2203.06173)
    """
    def __init__(
        self,
        input_channel=3,
        mvp_model_class='vitb-mae-egosoup',
        freeze=True,
    ):
        """
        Using MVP pretrained observation encoder network proposed by https://arxiv.org/abs/2203.06173
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            mvp_model_class (str): select one of the mvp pretrained model "vits-mae-hoi", "vits-mae-in", "vits-sup-in", "vitb-mae-egosoup" or "vitl-256-mae-egosoup"
            freeze (bool): if True, use a frozen MVP pretrained model.
        """
        super(MVPConv, self).__init__()

        try:
            import mvp
        except ImportError:
            print("WARNING: could not load mvp library! Please follow https://github.com/ir413/mvp to install MVP.")

        self.nets = mvp.load(mvp_model_class)
        if freeze:
            self.nets.freeze()

        assert input_channel == 3 # MVP only support input image with channel size 3
        assert mvp_model_class in ["vits-mae-hoi", "vits-mae-in", "vits-sup-in", "vitb-mae-egosoup", "vitl-256-mae-egosoup"] # make sure the selected r3m model do exist

        self._input_channel = input_channel
        self._freeze = freeze
        self._mvp_model_class = mvp_model_class
        self._input_coord_conv = False
        self._pretrained = True

        if '256' in mvp_model_class:
            input_img_size = 256
        else:
            input_img_size = 224
        self.preprocess = nn.Sequential(
            transforms.Resize(input_img_size)
        )

    def forward(self, inputs):
        x = self.preprocess(inputs)
        x = self.nets(x)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )
        return x

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        if 'vitb' in self._mvp_model_class:
            output_shape = [768]
        elif 'vitl' in self._mvp_model_class:
            output_shape = [1024]
        else:
            output_shape = [384]
        return output_shape

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={}, pretrained={}, freeze={})'.format(self._input_channel, self._input_coord_conv, self._pretrained, self._freeze)


class CoordConv2d(Module):
    """
    2D Coordinate Convolution

    Source: An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    https://arxiv.org/abs/1807.03247
    (e.g. adds 2 channels per input feature map corresponding to (x, y) location on map)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        coord_encoding='position',
    ):
        """
        Args:
            in_channels: number of channels of the input tensor [C, H, W]
            out_channels: number of output channels of the layer
            kernel_size: convolution kernel size
            stride: conv stride
            padding: conv padding
            dilation: conv dilation
            groups: conv groups
            bias: conv bias
            padding_mode: conv padding mode
            coord_encoding: type of coordinate encoding. currently only 'position' is implemented
        """

        assert(coord_encoding in ['position'])
        self.coord_encoding = coord_encoding
        if coord_encoding == 'position':
            in_channels += 2  # two extra channel for positional encoding
            self._position_enc = None  # position encoding
        else:
            raise Exception("CoordConv2d: coord encoding {} not implemented".format(self.coord_encoding))
        self.layers=Conv2d(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # adds 2 to channel dimension
        return [input_shape[1] + 2] + input_shape[2:]

    def forward(self, input):
        b, c, h, w = input.shape
        if self.coord_encoding == 'position':
            if self._position_enc is None:
                pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
                pos_y = pos_y.float().to(input.device) / float(h)
                pos_x = pos_x.float().to(input.device) / float(w)
                self._position_enc = torch.stack((pos_y, pos_x)).unsqueeze(0)
            pos_enc = self._position_enc.expand(b, -1, -1, -1)
            input = torch.cat((input, pos_enc), dim=1)
        return super(CoordConv2d, self).forward(input)

class ShallowConv(Module):
    """
    A shallow convolutional encoder from https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(self, input_channel=3, output_channel=32):
        super(ShallowConv, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.layers = nn.Sequential(
            Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3),
            ReLU(inplace=True),
            Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            ReLU(),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 4)
        assert(input_shape[1] == self.input_channel)
        out_h = int(math.floor(input_shape[2] / 2.))
        out_w = int(math.floor(input_shape[3] / 2.))
        return [input_shape[0],self.output_channel, out_h, out_w]
    

class Conv1d(Module):
    """
    Base class for stacked Conv1d layers.

    Args:
        input_channel (int): Number of channels for inputs to this network
        activation (None or str): Per-layer activation to use. Defaults to "relu". Valid options are
            currently {relu, None} for no activation
        out_channels (list of int): Output channel size for each sequential Conv1d layer
        kernel_size (list of int): Kernel sizes for each sequential Conv1d layer
        stride (list of int): Stride sizes for each sequential Conv1d layer
        conv_kwargs (dict): additional nn.Conv1D args to use, in list form, where the ith element corresponds to the
            argument to be passed to the ith Conv1D layer.
            See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html for specific possible arguments.
    """
    def __init__(
        self,
        input_channel=1,
        activation="relu",
        out_channels=(32, 64, 64),
        kernel_size=(8, 4, 2),
        stride=(4, 2, 1),
        **conv_kwargs
    ):
        super().__init__()

        assert activation in activation.keys(),"activation must be implemented"

        # Get activation requested
        activation = activation_map[activation]

        # Generate network
        self.n_layers = len(out_channels)
        self.layers = OrderedDict()
        for i in range(self.n_layers):
            self.layers[f'conv{i}'] = Conv1d(
                in_channels=input_channel,
                out_channels=out_channels[i],
                kernel_size=kernel_size[i],
                stride=stride[i]
            )
            if activation is not None:
                self.layers[f'act{i}'] = activation_map(activation)
            input_channel = out_channels[i]

        # Store network
        self.layers = Sequential(self.layers)

    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        output_shape=self.layers.output_shape(input_shape=input_shape)
        
        return output_shape


# %%
class ConvTranspose2d(Module):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:Union[int,Tuple[int,int]],stride:Union[int,Tuple[int,int]]=1,padding:Union[int,str,Tuple[int,int]]=0,**kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.layers=nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=self.padding,**kwargs)
    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        if isinstance(self.kernel_size,int):
            self.kernel_size=[self.kernel_size]*2
        if isinstance(self.stride,int):
            self.stride=[self.stride]*2
        if isinstance(self.padding,str):
            if self.padding == 'same':
                return input_shape
            elif self.padding == 'valid':
                self.padding = [0,0]
        elif isinstance(self.padding,int):
            self.padding=[self.padding]*2
        assert input_shape[1]==self.in_channels
        im_h= (input_shape[2]-1)*self.stride[0]-2*self.padding[0]+self.kernel_size[0]
        im_w = (input_shape[3]-1)*self.stride[1]-2*self.padding[1]+self.kernel_size[1]
        out_shape=[input_shape[0],self.out_channels,im_h,im_w]
        return out_shape
    def __repr__(self):
        """Pretty print network."""
        return f"{self.__class__.__name__}(input_channel={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride},padding={self.padding})"


# %%

# %%
#########
#Pooling#
#########

class MaxPool2d(Module):
    def __init__(self,kernel_size:Union[int, Tuple[int, int]],stride:Optional[Union[int, Tuple[int, int]]]=None, padding:Union[int, Tuple[int, int]]=0,**kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.layers = nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=padding,**kwargs)
    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        if isinstance(self.kernel_size,int):
            self.kernel_size=[self.kernel_size]*2
        if isinstance(self.stride,int):
            self.stride=[self.stride]*2
        if isinstance(self.padding,int):
            self.padding=[self.padding]*2
        im_h=(input_shape[2]+2*self.padding[0]-self.kernel_size[0])//self.stride[0]+1
        im_w=(input_shape[3]+2*self.padding[1]-self.kernel_size[1])//self.stride[1]+1
        return input_shape[:2]+[im_h,im_w]

class AvgPool2d(Module):
    def __init__(self,kernel_size:Union[int, Tuple[int, int]],stride:Optional[Union[int, Tuple[int, int]]]=None, padding:Union[int, Tuple[int, int]]=0,**kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.layers = nn.AvgPool2d(kernel_size=kernel_size,stride=stride,padding=padding,**kwargs)
    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        if isinstance(self.kernel_size,int):
            self.kernel_size=[self.kernel_size]*2
        if isinstance(self.stride,int):
            self.stride=[self.stride]*2
        if isinstance(self.padding,int):
            self.padding=[self.padding]*2
        im_h=(input_shape[2]+2*self.padding[0]-self.kernel_size[0])//self.stride[0]+1
        im_w=(input_shape[3]+2*self.padding[1]-self.kernel_size[1])//self.stride[1]+1
        return input_shape[:2]+[im_h,im_w]

class AdaptiveAvgPool2d(Module):
    def __init__(self,output_size:Union[int,Tuple[int,int]]):
        super().__init__()
        self.output_size = output_size
        self.layers = nn.AdaptiveAvgPool2d(output_size=output_size)
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        if isinstance(self.output_size,int):
            self.output_size=[self.output_size]*2
        return input_shape[:2]+list(self.output_size)

class MaxAvgPool2d(Module):
    def __init__(self, kernel_size:int,stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_pool = MaxPool2d(kernel_size, stride, padding)
        self.avg_pool = AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        max_output = self.max_pool(x)
        avg_output = self.avg_pool(x)
        combined_output = max_output + avg_output
        return combined_output
    
    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        if isinstance(self.kernel_size,int):
            self.kernel_size=[self.kernel_size]*2
        if isinstance(self.stride,int):
            self.stride=[self.stride]*2
        if isinstance(self.padding,int):
            self.padding=[self.padding]*2
        im_h=(input_shape[2]+2*self.padding[0]-self.kernel_size[0])//self.stride[0]+1
        im_w=(input_shape[3]+2*self.padding[1]-self.kernel_size[1])//self.stride[1]+1
        return input_shape[:2]+[im_h,im_w]


class SpatialSoftmax(Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape,
        num_kp=32,
        temperature=1.,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self.in_c, self.in_h, self.in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.layers = Conv2d(self.in_c, num_kp, kernel_size=1)
            self.num_kp = num_kp
        else:
            self.layers= None
            self.num_kp = self.in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
                )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        return f"{self.__class__.__name__}(num_kp={self.num_kp},self.temperature={self.temperature.item()},self.noise_std={self.noise_std})"

    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


class SpatialMeanPool(Module):
    """
    Module that averages inputs across all spatial dimensions (dimension 2 and after),
    leaving only the batch and channel dimensions.
    """
    def __init__(self, input_shape):
        super(SpatialMeanPool, self).__init__()
        assert len(input_shape) == 3 # [C, H, W]
        self.in_shape = input_shape

    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return list(self.in_shape[:1]) # [C, H, W] -> [C]

    def forward(self, inputs):
        """Forward pass - average across all dimensions except batch and channel."""
        return torch.flatten(inputs,start_dim=2).mean(dim=2)

class RNN(Module):
    def __init__(self,input_size:int,hidden_size:int,num_layers:int=1, nonlinearity:str='tanh',**kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.layers=nn.RNN(input_size=input_size, hidden_size=hidden_size, nonlinearity='tanh',**kwargs)
    def output_shape(self, input_shape:List[torch.Tensor])->Tuple[List[torch.Tensor],List[torch.Tensor]]:
        out_shape=input_shape
        out_shape[-1]=self.hidden_size
        hidden_shape=[self.num_layers,self.hidden_size]
        return (out_shape,hidden_shape)
    def forward(self,inputs:torch.Tensor,h:Optional[torch.Tensor]=None):
        if h is None:
            x,h_n=self.layers(inputs)
        else:
            x,h_n=self.layers(inputs,h)
        return x,h_n

    def __repr__(self):
        """Pretty print network."""
        return f"{self.__class__.__name__}(input_size={self.input_size},hidden_size={self.hidden_size},num_layers={self.num_layers},nonlinearity={self.nonlinearity})"

class GRU(Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int=1, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.layers = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **kwargs
        )

    def output_shape(self, input_shape:List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        out_shape = input_shape.copy()
        out_shape[-1] = self.hidden_size
        hidden_shape = [self.num_layers, self.hidden_size]
        return (out_shape, hidden_shape)

    def forward(self, inputs:torch.Tensor, h:Optional[torch.Tensor]=None):
        if h is None:
            x, h_n = self.layers(inputs)
        else:
            x, h_n = self.layers(inputs, h)
        return x, h_n

    def __repr__(self):
        return f"{self.__class__.__name__}(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers})"


class LSTM(Module):
    def __init__(self,input_size:int, hidden_size:int, num_layers:int=1,**kwargs):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.layers=nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,**kwargs)
    def output_shape(self, input_shape:List[torch.Tensor])->List[torch.Tensor]:
        out_shape=input_shape.copy()
        out_shape[-1]=self.hidden_size
        h_shape=input_shape.copy()
        h_shape[0],h_shape[-1]=self.num_layers,self.hidden_size
        c_shape=input_shape.copy()
        c_shape[0],c_shape[-1]=self.num_layers,self.hidden_size
        return out_shape,(h_shape,c_shape)
    def forward(self,inputs:torch.Tensor,hc:Optional[Tuple[torch.tensor,torch.tensor]]=None):
        if hc is not None:
            x,hc=self.layers(inputs,hc)
        else:
            x,hc=self.layers(inputs) 
        return x,hc
    def __repr__(self):
        """Pretty print network."""
        return f"{self.__class__.__name__}(input_size={self.input_size},hidden_size={self.hidden_size},num_layers={self.num_layers})"

# class MaxAvgPooling(nn.Module):
#     def __init__(self, kernel_size, stride=None, padding=0):
#         super(MaxAvgPooling, self).__init__()
#         self.max_pool = MaxPool2d(kernel_size, stride, padding)
#         self.avg_pool = AvgPool2d(kernel_size, stride, padding)
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
    
#     def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
#         if isinstance(self.kernel_size,int):
#             self.kernel_size=[self.kernel_size]*2
#         if isinstance(self.stride,int):
#             self.stride=[self.stride]*2
#         if isinstance(self.padding,int):
#             self.padding=[self.padding]*2
#         im_h=(input_shape[2]+2*self.padding[0]-self.kernel_size[0])//self.stride[0]+1
#         im_w=(input_shape[3]+2*self.padding[1]-self.kernel_size[1])//self.stride[1]+1
#         return input_shape[:2]+[im_h,im_w]

#     def forward(self, x):
#         max_output = self.max_pool(x)
#         avg_output = self.avg_pool(x)
#         combined_output = max_output + avg_output
#         return combined_output

# %%
class ResNet18Conv(Module):
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super().__init__()
        net = vision_models.resnet18(pretrained=pretrained)

        if input_coord_conv:
            net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            net.conv1 = Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self.input_coord_conv = input_coord_conv
        self.input_channel = input_channel
        self.layers = nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape:List[torch.Size])-> List[torch.Size]:
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 4)
        out_h = int(math.ceil(input_shape[2] / 32.))
        out_w = int(math.ceil(input_shape[3] / 32.))
        return [input_shape[0],512, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        return f"{self.__class__.__name__}(input_channel={self.input_channel}, input_coord_conv={self.input_coord_conv})"
    

# %%
class OverlapPatchMerging(Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        self.cn1 = Conv2d(in_channels, out_channels, kernel_size=patch_size, stride = stride, padding = padding)
        self.flatten = Flatten(start_dim=2)
        self.layerNorm = LayerNorm(out_channels)
    
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        out_shape=self.cn1.output_shape(input_shape)
        B,C,H,W = out_shape
        return [B,C,H*W]
        

    def forward(self, patches):
        """Merge patches to reduce dimensions of input.

        :param patches: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, C, H, W)
        """
        x = self.cn1(patches)
        _,_,H, W = x.shape
        x = self.flatten(x)  # B，C，N(HW)
        x = x.transpose(1,2) #B,N(HW),C
        x = self.layerNorm(x)
        return x,H,W #B, N, EmbedDim
    
class EfficientSelfAttention(Module):
    def __init__(self, channels, reduction_ratio, num_heads):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} should be divided by num_heads {num_heads}."
        self.num_heads= num_heads
        self.dim_heads = channels // num_heads
        self.scale = self.dim_heads ** 0.5
        #### Self Attention Block consists of 2 parts - Reduction and then normal Attention equation of queries and keys###
        
        # Reduction Parameters #
        self.cn1 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.ln1 = LayerNorm(normalized_shape=channels)
        # Attention Parameters #
        self.keyValueExtractor = Linear(input_dim=channels, output_dim=channels * 2)
        self.query = Linear(input_dim=channels, output_dim=channels)
        self.smax = Softmax(dim=-1)
        self.finalLayer = Linear(input_dim=channels, output_dim=channels) 

    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def forward(self,inputs,H,W):

        """ Perform self attention with reduced sequence length

        :param x: tensor of shape (B, N, C) where
            B is the batch size,
            N is the number of queries (equal to H * W)
            C is the number of channels
        :return: tensor of shape (B, N, C)
        """
        x = inputs  #(B,N,C)
        B,N,C = x.shape  
        x1 = x.clone().permute(0,2,1)  #(B,N,C)->(B,C,N) 
        x1 = x1.reshape(B,C,H,W)  #(B,C,N)->(B,C,H,W)
        x1 = self.cn1(x1)  #(B,C,H,W)->(B,C,H//reduction,W//reduction)
        x1 = x1.reshape(B,C,-1).permute(0,2,1) #(B,H//reduction*W//reduction,C)
        x1 = self.ln1(x1) 
        keyVal = self.keyValueExtractor(x1)
        keyVal = keyVal.reshape(B, -1 , 2, self.num_heads, self.dim_heads).permute(2,0,3,1,4)
        k,v = keyVal[0],keyVal[1] #b,heads, n, c/heads
        q = self.query(x).reshape(B, N, self.num_heads, self.dim_heads).permute(0, 2, 1, 3)
        attention = self.smax(q@k.transpose(-2, -1)/self.scale)  ## (QK.T)/sqrt(d),(B,num_heads,H//reduction*W//reduciton,H//reduction*W//reduciton)
        attention = (attention@v).transpose(1,2).reshape(B,-1,C) #(B,num_heads,H//reduction*W//reduciton,dim_heads)->(B,H//reduction*W//reduction,num_heads,dim_heads)->(B,N,C)
        x = self.finalLayer(attention)  #(B,N,C)
        return x

class MixFFN(Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        expanded_channels = channels*expansion_factor
        #MLP Layer        
        self.mlp1 = Linear(input_dim=channels, output_dim=expanded_channels)
        #Depth Wise CNN Layer
        self.conv= Conv2d(in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=3,  padding='same', groups=channels)
        #GELU
        self.gelu = GELU()
        #MLP to predict
        self.mlp2 = Linear(input_dim=expanded_channels, output_dim=channels)
    def output_shape(self,input_shape:List[torch.Size])->torch.Size:
        return input_shape
    def forward(self, inputs,H,W):
        """ Perform self attention with reduced sequence length

        :param x: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, C, H, W)
        """
        x = inputs     #B,N,C
        x = self.mlp1(x) #B,N,C*exp
        B,N,C = x.shape
        x = x.transpose(1,2).reshape(B,C,H,W) #B,C*exp,H,W

        #Depth Conv - B, N, Cexp 
        x = self.gelu(self.conv(x).flatten(2).transpose(1,2)) #(B,N,C*exp)

        #Back to the orignal shape
        x = self.mlp2(x) # BNC
        return x

class MixTransformerEncoderLayer(Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding, 
                n_layers, reduction_ratio, num_heads, expansion_factor):
        super().__init__()
        self.n_layers = n_layers
        self.patchMerge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding) # B N embed dim
        self._attn = nn.ModuleList([EfficientSelfAttention(out_channels, reduction_ratio, num_heads) for _ in range(n_layers)])
        self._ffn = nn.ModuleList([MixFFN(out_channels,expansion_factor) for _ in range(n_layers)])
        self._lNorm = nn.ModuleList([LayerNorm(out_channels) for _ in range(n_layers)])
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape
    def forward(self, inputs):
        """ Run one block of the mix vision transformer

        :param x: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, C, H, W)
        """
        x = inputs
        B,_,_,_ = x.shape
        x,H,W = self.patchMerge(x) # B N C
        for i in range(self.n_layers):
            x = x + self._attn[i](x, H, W) #BNC
            x = x + self._ffn[i](x,H,W) #BNC
            x = self._lNorm[i](x) #BNC
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2) #BCHW
        return x 
    
class SelfAttentionCl(Module):
    """ Channels-last multi-head self-attention (B, ..., C)
        self-attention 通常对最后一个通道维度进行变换转换为d
    """ 
    def __init__(
            self,
            dim: int,
            dim_head: int = 10,
            bias: bool = True):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=bias)
        self.proj = Linear(dim, dim, bias=bias)
    
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3) #切分成三份 B,..,num_head,dim_head

        attn = (q @ k.transpose(-2, -1)) * self.scale   #B,..,num_head,num_head
        attn = attn.softmax(dim=-1) #B,..,num_head,num_head

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,)) #B,..,num_head,dim_head ->B,..,C
        x = self.proj(x)  #B,..,C
        return x

class LayerScale(Module):
    def __init__(self, dim: int, init_values: float=1e-5, inplace: bool=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return input_shape

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma
    
class Block_SA(Module):
    def __init__(self,dim:int,window_size:Tuple[int,int]):  #dim 表示通道维度
        super().__init__()
        self.window_size=window_size
        self.atten = SelfAttentionCl(dim=dim)  
        self.ls = LayerScale(dim=dim,init_values=1e-5)
        self.mlp = [
            Linear(input_dim=dim,output_dim=dim*4),
            ReLU(),
            Linear(input_dim=dim*4, output_dim=dim, bias=True),
            ReLU(),
        ]
        self.mlp=nn.Sequential(*self.mlp) 
    def forward(self,x):
        x = F.normalize(x,dim=-1)   
        B,H,W,C = x.shape
        x = x.reshape(B, H // self.window_size[0], self.window_size[0], W // self.window_size[1], self.window_size[1], C)
        windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size[0], self.window_size[1], C)
        windows = windows + self.atten(windows)
        x = windows.reshape(-1, H // self.window_size[0], W // self.window_size[1], self.window_size[0], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        x = F.normalize(x,dim=-1)
        x = x +self.ls(self.mlp(x))
        return x

class Grid_SA(Module):
    def __init__(self,dim:int,grid_size:Tuple[int,int]):
        super().__init__()
        self.grid_size=grid_size
        self.atten = SelfAttentionCl(dim=dim)
        self.ls = LayerScale(dim=dim,init_values=1e-5)
        self.mlp = [
            Linear(input_dim=dim,output_dim=dim*4),
            ReLU(),
            Linear(input_dim=dim*4, output_dim=dim, bias=True),
        ]
        self.mlp=nn.Sequential(*self.mlp)

    def forward(self,x):
        x = F.normalize(x,dim=-1)
        B,H,W,C = x.shape
        x = x.view(B, self.grid_size[0], H // self.grid_size[0], self.grid_size[1], W // self.grid_size[1], C)
        grids = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, self.grid_size[0], self.grid_size[1], C)
        grids = grids + self.atten(grids)
        x = grids.reshape(-1, H // self.grid_size[0], W // self.grid_size[1], self.grid_size[0], self.grid_size[1], C)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, H, W, C)
        x = F.normalize(x,dim=-1)
        x = x +self.ls(self.mlp(x))
        return x
#%%
class DoubleConv(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = Sequential(
            # 设置 padding=1 以保持特征图尺寸不变
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            # 设置 padding=1 以保持特征图尺寸不变
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class UNetEncoder(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = Sequential(
            MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)

class UNetDecoder(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3,recurrent:int=0):
        super().__init__()
        self.recurrent = recurrent
        self.in_conv = DoubleConv(in_channels, 64)
        self.encoder1 = UNetEncoder(64, 128)
        self.encoder2 = UNetEncoder(128, 256)
        self.encoder3 = UNetEncoder(256, 512)
        self.encoder4 = UNetEncoder(512, 1024)
        if self.recurrent:
            self.lstm =LSTM(input_size=15,hidden_size=15,batch_first=True)
        self.decoder1 = UNetDecoder(1024, 512)
        self.decoder2 = UNetDecoder(512, 256)
        self.decoder3 = UNetDecoder(256, 128)
        self.decoder4 = UNetDecoder(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, hc:Optional[Tuple[torch.Tensor,torch.Tensor]]=None):
        x1 = self.in_conv(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        hc = None
        if self.recurrent:
            b,c,h,w = x5.shape
            x5 = x5.flatten(start_dim=2)
            if hc is not None:
                x5,hc = self.lstm(x5,hc)
            else:
                x5,hc = self.lstm(x5)
            x5 = x5.reshape(b,c,h,w)
        x = self.decoder1(x5, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)
        x = self.out_conv(x)
        return x,hc

#%%
class Res_block(Module):
    def __init__(self,in_channels:int,out_channels:int,stride:int=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bottleneck = Sequential(
            BatchNorm2d(in_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=False),
        )
        self.downsample = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
            BatchNorm2d(out_channels),
        )
        self.relu = ReLU()
    def output_shape(self,input_size:List[torch.Size])->List[torch.Size]:
        B,C,H,W = input_size
        assert C == self.in_channels
        output_size = [B,self.out_channels,ceil(H/self.stride),ceil(W/self.stride)]
        return output_size
    def forward(self,x):
        out = self.bottleneck(x)
        x_d = self.downsample(x)
        out = self.relu(out + x_d)
        if self.output_shape(list(x.shape))!=list(out.shape):
            raise ValueError(f'the ideal size is {self.output_shape(list(x.shape))} but got the size {list(out.shape)}')
        else:
            return out
        
class RVT_block(Module):
    def __init__(self, is_downsample: bool = True, window_size: Tuple[int, int] = (3,3), 
                input_shape: Tuple[torch.Size] = None, downsample_channels: int = 20,
                conv_kernel_size: int = 3,conv_padding_size:int=1,pool_kernel_size: int = 2, pool_stride: int = 2,use_lstm:bool=False):
        super().__init__()
        self.use_lstm = use_lstm
        self.is_downsample = is_downsample
        _, C, H, W = input_shape
        if self.is_downsample:
            self.downsample = Sequential(
                Conv2d(in_channels=C, out_channels=downsample_channels, kernel_size=conv_kernel_size, padding=conv_padding_size),
                MaxAvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            )
            C = downsample_channels
            if self.use_lstm:
                self.lstm = LSTM(input_size=int(H / pool_kernel_size * W / pool_kernel_size), 
                                hidden_size=int(H / pool_kernel_size * W / pool_kernel_size))
        else:
            if self.use_lstm:
                self.lstm = LSTM(input_size=int(H * W), hidden_size=int(H * W))
        self.block_sa = Block_SA(dim=C, window_size=window_size)
        self.grid_sa = Grid_SA(dim=C, grid_size=window_size)
        self.ls = LayerScale(dim=C, init_values=1e-5)

    def output_shape(self, input_shape: List[int]) -> List[int]:
        if self.is_downsample:
            return self.downsample.output_shape(input_shape)
        else:
            return input_shape

    def forward(self, inputs: torch.Tensor, h_and_c_previous: Optional[torch.Tensor] = None):
        x = inputs
        if self.is_downsample:
            x = self.downsample(x)
        B, C, H, W = x.shape
        x = torch.einsum("bchw->bhwc", x)
        x = self.block_sa(x)
        x = self.grid_sa(x)
        x = torch.einsum("bhwc->bchw", x)
        if self.use_lstm:
            x = x.reshape(B, C, H * W)
            x, h_c_tuple = self.lstm(x, h_and_c_previous)
            x = x.reshape(B, C, H, W)
        #x = h_c_tuple[0]
            return x,h_c_tuple
        else :
            return x

class Fusion_block(Module):
    def __init__(self,dvs_shape:List[int],depth_shape:List[int],out_channels:int,stride:int,local_rank:int=None):
        super().__init__()
        assert dvs_shape[2:] == depth_shape[2:]
        H,W = dvs_shape[2:]
        self.out_channels = out_channels
        self.stride = stride
        self.res_block = Res_block(in_channels=depth_shape[1],out_channels=out_channels,stride=stride)
        self.rvt_block = RVT_block(downsample_channels=out_channels,input_shape=dvs_shape,pool_kernel_size=stride,pool_stride=stride)
        self.linear = Linear(input_dim=ceil(H/stride)*ceil(W/stride)*2*out_channels,output_dim=2)
        self.softmax = Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.randn(1))
        if local_rank is not None:
            device = torch.device(f'cuda:{local_rank}')
            self = self.to(device)
    def output_shape(self,dvs_shape:List[int],depth_shape:List[int]):
        assert dvs_shape[2:] == depth_shape[2:]
        B,_,H,W = dvs_shape
        return [B,self.out_channels,ceil(H/self.stride),ceil(W/self.stride)],[B,self.out_channels,ceil(H/self.stride),ceil(W/self.stride)]
    def forward(self,dvs,depth):
        dvs_f = self.rvt_block(dvs)
        depth_f = self.res_block(depth)
        feature = torch.cat([dvs_f,depth_f],dim=1)
        feature = feature.flatten(start_dim=1)
        coff = self.linear(feature)
        alpha,beta = self.softmax(coff+torch.randn_like(coff)*self.sigma).unbind(dim=-1)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        fusion_f = beta*depth_f + alpha*dvs_f
        return fusion_f,dvs_f,depth_f
    
class Bi_CrossAttention(Module):
    def __init__(self,indim1,indim2,k_dim,v_dim,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.k1_linear = nn.Linear(in_features=indim1,out_features=k_dim*num_heads)
        self.k2_linear = nn.Linear(in_features=indim2,out_features=k_dim*num_heads)
        self.v1_linear = nn.Linear(in_features=indim1,out_features=v_dim*num_heads)
        self.v2_linear = nn.Linear(in_features=indim2,out_features=v_dim*num_heads)
        self.o1_linear = nn.Linear(in_features=v_dim*num_heads,out_features=indim1)
        self.o2_linear = nn.Linear(in_features=v_dim*num_heads,out_features=indim2)
    def forward(self,input1:torch.Tensor,input2:torch.Tensor):
        B1,C1,H1,W1 = input1.shape
        B2,C2,H2,W2 = input2.shape
        input1_embedding = input1.reshape(B1,C1,H1*W1).permute(0,2,1)
        input2_embedding = input2.reshape(B2,C2,H2*W2).permute(0,2,1)
        k1 = self.k1_linear(input1_embedding).reshape(B1,H1*W1,self.num_heads,self.k_dim).permute(0,2,1,3)
        k2 = self.k2_linear(input2_embedding).reshape(B2,H2*W2,self.num_heads,self.k_dim).permute(0,2,1,3)
        v1 = self.v1_linear(input1_embedding).reshape(B1,H1*W1,self.num_heads,self.v_dim).permute(0,2,1,3)
        v2 = self.v2_linear(input2_embedding).reshape(B2,H2*W2,self.num_heads,self.v_dim).permute(0,2,1,3)
        o1 = torch.einsum('...nqd,...nkd->...nqk',k1,k2)/sqrt(self.k_dim)
        o2 = torch.einsum('...nkd,...nqd->...nkq',k2,k1)/sqrt(self.k_dim)
        o1 = F.softmax(o1,dim=-1)
        o2 = F.softmax(o2,dim=-1)
        output1 = torch.matmul(o1,v2).permute(0,2,1,3).reshape(B1,H1*W1,-1)
        output1 = self.o1_linear(output1).permute(0,2,1).reshape(B1,C1,H1,W1)
        output2 = torch.matmul(o2,v1).permute(0,2,1,3).reshape(B1,H2*W2,-1)
        output2 = self.o2_linear(output2).permute(0,2,1).reshape(B2,C2,H2,W2)
        return output1,output2

class CrossAttention(nn.Module):
    def __init__(self,indim1,indim2,k_dim,v_dim,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.k1_linear = Linear(input_dim=indim1,output_dim=k_dim*num_heads)
        self.k2_linear = Linear(input_dim=indim2,output_dim=k_dim*num_heads)
        self.v2_linear = Linear(input_dim=indim2,output_dim=v_dim*num_heads)
        self.o1_linear = Linear(input_dim=v_dim*num_heads,output_dim=indim1)
    def forward(self,input1:torch.Tensor,input2:torch.Tensor):
        B1,C1,H1,W1 = input1.shape
        B2,C2,H2,W2 = input2.shape
        input1_embedding = input1.reshape(B1,C1,H1*W1).permute(0,2,1)
        input2_embedding = input2.reshape(B2,C2,H2*W2).permute(0,2,1)
        k1 = self.k1_linear(input1_embedding).reshape(B1,H1*W1,self.num_heads,self.k_dim).permute(0,2,1,3)
        k2 = self.k2_linear(input2_embedding).reshape(B2,H2*W2,self.num_heads,self.k_dim).permute(0,2,1,3)
        v2 = self.v2_linear(input2_embedding).reshape(B2,H2*W2,self.num_heads,self.v_dim).permute(0,2,1,3)
        o1 = torch.einsum('...nqd,...nkd->...nqk',k1,k2)/sqrt(self.k_dim)
        o1 = F.softmax(o1,dim=-1)
        o1 = torch.nan_to_num(o1, nan=0.0)
        output1 = torch.matmul(o1,v2).permute(0,2,1,3).reshape(B1,H1*W1,-1)
        output1 = self.o1_linear(output1).permute(0,2,1).reshape(B1,C1,H1,W1)
        return output1

class FPN_PAN(Module):
    def __init__(self, in_channels_list):
        super(FPN_PAN, self).__init__()
        # FPN 部分
        self.fpn_laterals = nn.ModuleList([
            nn.Conv2d(in_channels, 60, kernel_size=1) for in_channels in in_channels_list
        ])
        self.fpn_topdowns = nn.ModuleList([
            nn.Conv2d(60, 60, kernel_size=3, padding=1) for _ in range(len(in_channels_list) - 1)
        ])
        # PAN 部分
        self.pan_bottomups = nn.ModuleList([
            nn.Conv2d(60, 60, kernel_size=3, stride=2, padding=1) for _ in range(len(in_channels_list) - 1)
        ])

    def forward(self, x1, x2, x3):
        features = [x1, x2, x3]
        # FPN 自顶向下路径
        laterals = [lateral(feature) for lateral, feature in zip(self.fpn_laterals, features)]
        fpn_features = [laterals[-1]]
        for i in range(len(laterals) - 2, -1, -1):
            if i == len(laterals) - 2:
                # x3 到 x2 尺寸相同，无需上采样
                topdown = fpn_features[-1]
            else:
                # x2 到 x1 上采样 3 倍
                topdown = nn.functional.interpolate(fpn_features[-1], scale_factor=2, mode='bilinear', align_corners=True)
            fpn_features.append(self.fpn_topdowns[i](topdown + laterals[i]))
        fpn_features.reverse()

        # PAN 自底向上路径
        pan_features = [fpn_features[0]]
        for i in range(len(fpn_features) - 1):
            if i == 0:
                # x1 到 x2 下采样 3 倍
                bottomup = self.pan_bottomups[i](pan_features[-1])
            else:
                # x2 到 x3 尺寸相同，无需下采样
                bottomup = pan_features[-1]
            pan_features.append(bottomup + fpn_features[i + 1])

        return pan_features
    
