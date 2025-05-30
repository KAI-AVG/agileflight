# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional,Tuple
import os
base_path=os.environ.get("project_path")
from ConvLSTM_pytorch.convlstm import ConvLSTM
from base_model import *
from typing import List
from omegaconf import DictConfig
from numpy import prod
# %%
def center_crop(tensor: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """Crop tensor to target size from center."""
    _, _, h, w = tensor.shape
    th, tw = target_size
    i = (h - th) // 2
    j = (w - tw) // 2
    return tensor[:, :, i:i+th, j:j+tw]

activation_map = {'relu':ReLU(),'leaky_relu':Leaky_RELU(),'tanh':Tanh(),'sigmoid':Sigmoid(),'gelu':GELU()}
# %%
class BaseModel(Module):
    def __init__(self):
        super().__init__()
        #self.cfg = cfg
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# %%
class OrigUnet_lstm(BaseModel):
    def __init__(self,unet:DictConfig,conv:DictConfig,linear:DictConfig,input_mode:str='dvs',image_shape:List[int]=[60,90]):
        super().__init__()
        assert input_mode in ['dvs','depth'],"input mode must be dvs or depth"
        if input_mode == 'depth':
            self.unet = UNet(in_channels=1,out_channels=unet.out_channels,recurrent=unet.recurrent)
        else:
            self.unet = UNet(in_channels=2,out_channels=unet.out_channels,recurrent=unet.recurrent)
        self.conv = []
        in_channel = 1
        for channel,pool_stride,kernel,padding in zip(conv.channels,conv.pool_strides,conv.conv_kernels,conv.conv_paddings):
            self.conv += [
                Conv2d(in_channels=in_channel,out_channels=channel,kernel_size=kernel,padding=padding),
                BatchNorm2d(channel),
                MaxPool2d(kernel_size=pool_stride,stride=pool_stride)
            ]
            in_channel = channel
        H,W = image_shape[0]//prod(conv.pool_strides),image_shape[1]//prod(conv.pool_strides)
        self.conv=Sequential(*self.conv)
        in_dims = in_channel*H*W
        self.fc = []
        for dim,activation in zip(linear.dims,linear.activations):
            self.fc.append(Linear(input_dim=in_dims,output_dim=dim))
            if dim != linear.dims[-1]:
                self.fc.append(Dropout(p=linear.drop_p))
            assert activation in activation_map.keys(),"activation must be inplemented"
            self.fc.append(activation_map[activation])
            in_dims = dim
        self.fc=Sequential(*self.fc)
        self._init_weights()
    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        return [input_shape[0],2]
        #self.log_model_info()
    def forward(self,x,state=None,extras:Optional[Tuple[torch.Tensor,torch.Tensor]]=None):
        if extras is None:
            depth_r,hc = self.unet(x)
        else:
            depth_r,hc = self.unet(x,extras)
        x = self.conv(depth_r)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x,hc,depth_r


# %%
class Vit(Module):
    """
    ViT+LSTM Network 
    Num Params: 3,563,663   
    """
    def __init__(self,encoder:DictConfig,decoder:DictConfig,lstm:DictConfig,linear:DictConfig,upsample_size,upscale_factor,downsample,input_mode:str='dvs'):
        super().__init__()
        assert input_mode in ['dvs','depth'],"input mode must be dvs or depth"
        if input_mode == 'dvs':
            input_channel = 1
        else:
            input_channel = 2
        self.encoder_blocks = nn.ModuleList()
        for channel_dim,patch_size,stride,padding,n_layer,reduction_ratio,num_head,expansion_factor in zip(encoder.channel_dims,encoder.patch_sizes,encoder.strides,encoder.paddings,encoder.n_layers,encoder.reduction_ratios,encoder.num_heads,encoder.expansion_factors):
            self.encoder_blocks.append(MixTransformerEncoderLayer(in_channels=input_channel,out_channels=channel_dim,patch_size=patch_size,stride=stride,padding=padding,n_layers=n_layer,reduction_ratio=reduction_ratio,num_heads=num_head,expansion_factor=expansion_factor))
            input_channel = channel_dim
        self.decoder = Linear(input_dim=decoder.dims[0],output_dim=decoder.dims[1])
        self.lstm = LSTM(input_size=lstm.input_dim + 10, hidden_size=lstm.hidden_size,
                        num_layers=lstm.num_layers, dropout=lstm.p)
        input_dim = lstm.hidden_size
        self.mlp = []
        for dim,activation in zip(linear.dims,linear.activations):
            self.mlp.append(Linear(input_dim=input_dim,output_dim=dim))
            if dim != linear.dims[-1]:
                self.mlp.append(Dropout(p=linear.p))
                self.mlp.append(BatchNorm1d(dim))
            assert activation in activation_map.keys(),"activation must be inplemented"
            self.mlp.append(activation_map[activation])
            input_dim = dim
        self.mlp = Sequential(*self.mlp)
        self.up_sample = nn.Upsample(size=tuple(upsample_size), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.down_sample = Conv2d(in_channels=downsample[0],out_channels=downsample[1],kernel_size=3, padding = 1)
    
    def output_shape(self,input_shape:List[torch.Size])->List[torch.Size]:
        return [input_shape[0],2]

    def forward(self, inputs,states=None,extras=None):
        embeds = [inputs]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:] #所有处理过的patch
        out = torch.cat([self.pxShuffle(out[1]),self.up_sample(out[0])],dim=1)  #拼接所有特征
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))
        out = torch.cat((out,states),dim=1)
        if extras:
            out,hc = self.lstm(out,extras)
        else:
            out,hc = self.lstm(out)
        out = self.mlp(out)
        return out,hc,None


#%%
class RVT_Stack(Module):
    def __init__(self,image_shape:Tuple[int,int],rvt_configs,batch_size:int,conv:DictConfig,linear,input_mode:str='dvs'):
        super().__init__()
        assert input_mode in ['dvs','depth'],"input mode must be in dvs or depth"
        self.rvt_blocks = nn.ModuleList()
        if input_mode == 'dvs':
            current_shape = [batch_size,2]+image_shape
        elif input_mode == 'depth':
            current_shape = [batch_size,1]+image_shape
        for is_downsample, kwargs in rvt_configs:
            rvt = RVT_block(is_downsample=is_downsample, input_shape=current_shape, **kwargs)
            self.rvt_blocks.append(rvt)
            current_shape = rvt.output_shape(current_shape)
        B,C,H,W = current_shape
        self.conv = Conv2d(in_channels=C,out_channels=conv.out_channels,kernel_size=conv.kernel_size)
        input_dim = conv.out_channels * H * W + 10
        self.mlp = []
        for dim,activaction in zip(linear.dims,linear.activations):
            self.mlp.append(Linear(input_dim=input_dim,output_dim=dim))
            if dim != linear.dims[-1]:
                self.mlp.append(BatchNorm1d(dim))
                self.mlp.append(Dropout(linear.p))
            self.mlp.append(activation_map[activaction])
            input_dim = dim
        self.mlp = Sequential(*self.mlp)
    def output_shape(self, input_shape:List[torch.Size])->List[torch.Size]:
        out_shape=[input_shape[0],2]
        return out_shape
    def forward(self,inputs,state,extras):
        others = None
        x = inputs
        for rvt in self.rvt_blocks:
            x = rvt(x)
        x = self.conv(x).flatten(start_dim=1)
        x = torch.cat((x,state),dim=1)
        vel_predict = self.mlp(x)
        return vel_predict,extras,others
#%%
class FusionNet(Module):
    def __init__(self,batch_size,image_shape,rvt,attention,rnn_type,rnn,linear,cross_mode):
        super().__init__()
        assert cross_mode in ["dvs2depth","depth2dvs"]
        self.cross_mode = cross_mode
        depth_shape = [batch_size,1]+image_shape
        dvs_shape = [batch_size,2]+image_shape
        rvt_config = rvt
        self.rnn_type = rnn_type
        self.rvt1 = []
        self.rvt2 = []
        self.mlp = []
        for channel,stride in rvt_config:
            rvt1=RVT_block(downsample_channels=channel,pool_kernel_size=stride,pool_stride=stride,use_lstm=False,input_shape=depth_shape)
            self.rvt1.append(rvt1)
            depth_shape = rvt1.output_shape(depth_shape)
            rvt2=RVT_block(downsample_channels=channel,pool_kernel_size=stride,pool_stride=stride,use_lstm=False,input_shape=dvs_shape)
            self.rvt2.append(rvt2)
            dvs_shape = rvt2.output_shape(dvs_shape)
        self.rvt1 = Sequential(*self.rvt1)
        self.rvt2 = Sequential(*self.rvt2)
        _,C,H,W = dvs_shape
        self.crossattention = CrossAttention(indim1=C,indim2=C,k_dim=attention.k_dim,v_dim=attention.v_dim,num_heads=attention.num_head)
        input_dims = C*H*W+10
        if self.rnn_type == 'lstm':
            self.rnn = LSTM(input_size=input_dims,hidden_size=input_dims,num_layers=rnn.num_layers)
        elif self.rnn_type == 'gru':
            self.rnn = GRU(input_size=input_dims, hidden_size=input_dims,
                            num_layers=rnn.num_layers)
        elif self.rnn_type == 'rnn':
            self.rnn = RNN(input_size=input_dims, hidden_size=input_dims,
                            num_layers=rnn.num_layers)
        for dim,activation in zip(linear.dims,linear.activations):
            self.mlp.append(Linear(input_dim=input_dims,output_dim=dim))
            if dim!=linear.dims[-1]:
                self.mlp.append(BatchNorm1d(num_feature=dim))
                self.mlp.append(Dropout(p=0.3))
            self.mlp.append(activation_map[activation])
            input_dims = dim
        self.mlp = Sequential(*self.mlp)
    #     self._init_weights()
    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    def forward(self,dvs,depth,state,extras=None):
        depth_f = self.rvt1(depth)
        dvs_f = self.rvt2(dvs)
        if self.cross_mode == 'dvs2depth':
            feature = self.crossattention(dvs_f,depth_f)
        else:
            feature = self.crossattention(depth_f,dvs_f)
        feature = feature.flatten(start_dim=1)
        hc = None
        x = torch.cat((feature,state),dim=1)
        if self.rnn_type is not None:
            x,hc = self.rnn(x,extras)
        vel = self.mlp(x)
        return vel,hc,None
    

#%%
class FusionCross(Module):
    def __init__(self,batch_size,image_shape,rvt,has_state,attention,rnn_type,rnn,linear):
        super().__init__()
        depth_shape = [batch_size,1]+image_shape
        dvs_shape = [batch_size,2]+image_shape
        rvt_config = rvt
        self.rnn_type = rnn_type
        self.has_state = has_state
        self.rvt1 = []
        self.rvt2 = []
        self.mlp = []
        for channel,stride in rvt_config:
            rvt1=RVT_block(downsample_channels=channel,pool_kernel_size=stride,pool_stride=stride,use_lstm=False,input_shape=depth_shape)
            self.rvt1.append(rvt1)
            depth_shape = rvt1.output_shape(depth_shape)
            rvt2=RVT_block(downsample_channels=channel,pool_kernel_size=stride,pool_stride=stride,use_lstm=False,input_shape=dvs_shape)
            self.rvt2.append(rvt2)
            dvs_shape = rvt2.output_shape(dvs_shape)
        self.rvt1 = Sequential(*self.rvt1)
        self.rvt2 = Sequential(*self.rvt2)
        _,C,H,W = dvs_shape
        self.crossattention = Bi_CrossAttention(indim1=C,indim2=C,k_dim=attention.k_dim,v_dim=attention.v_dim,num_heads=attention.num_head)
        if has_state:
            input_dims = C*H*W+10
        else:
            input_dims = C*H*W
        if self.rnn_type == 'lstm':
            self.rnn = LSTM(input_size=input_dims,hidden_size=input_dims,num_layers=rnn.num_layers)
        elif self.rnn_type == 'gru':
            self.rnn = GRU(input_size=input_dims, hidden_size=input_dims,
                            num_layers=rnn.num_layers)
        elif self.rnn_type == 'rnn':
            self.rnn = RNN(input_size=input_dims, hidden_size=input_dims,
                            num_layers=rnn.num_layers)
        for dim,activation in zip(linear.dims,linear.activations):
            self.mlp.append(Linear(input_dim=input_dims,output_dim=dim))
            if dim!=linear.dims[-1]:
                self.mlp.append(BatchNorm1d(num_feature=dim))
                self.mlp.append(Dropout(p=0.3))
            self.mlp.append(activation_map[activation])
            input_dims = dim
        self.mlp = Sequential(*self.mlp)
    #     self._init_weights()
    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    def forward(self,dvs,depth,state,extras=None):
        depth_f = self.rvt1(depth)
        dvs_f = self.rvt2(dvs)
        out1,out2 = self.crossattention(dvs_f,depth_f)
        feature = out1 + out2
        feature = feature.flatten(start_dim=1)
        hc = None
        if self.has_state:
            x = torch.cat((feature,state),dim=1)
        if self.rnn_type is not None:
            x,hc = self.rnn(x,extras)
        vel = self.mlp(x)
        return vel,hc,None
    
class ResCross(Module):
    def __init__(self,batch_size,image_shape,rvt,has_state,attention,rnn_type,rnn,linear):
        super().__init__()
        depth_shape = [batch_size,1]+image_shape
        dvs_shape = [batch_size,2]+image_shape
        rvt_config = rvt
        self.rnn_type = rnn_type
        self.has_state = has_state
        self.rvt1 = []
        self.rvt2 = []
        self.mlp = []
        in_channel1 = 1
        in_channel2 = 2
        for channel,stride in rvt_config:
            rvt1=Res_block(in_channels=in_channel1,out_channels=channel,stride=stride)
            self.rvt1.append(rvt1)
            depth_shape = rvt1.output_shape(depth_shape)
            rvt2=Res_block(in_channels=in_channel2,out_channels=channel,stride=stride)
            self.rvt2.append(rvt2)
            dvs_shape = rvt2.output_shape(dvs_shape)
            in_channel1 = channel
            in_channel2 = channel
        self.rvt1 = Sequential(*self.rvt1)
        self.rvt2 = Sequential(*self.rvt2)
        _,C,H,W = dvs_shape
        self.crossattention = Bi_CrossAttention(indim1=C,indim2=C,k_dim=attention.k_dim,v_dim=attention.v_dim,num_heads=attention.num_head)
        if has_state:
            input_dims = C*H*W+10
        else:
            input_dims = C*H*W
        if self.rnn_type == 'lstm':
            self.rnn = LSTM(input_size=input_dims,hidden_size=input_dims,num_layers=rnn.num_layers)
        elif self.rnn_type == 'gru':
            self.rnn = GRU(input_size=input_dims, hidden_size=input_dims,
                            num_layers=rnn.num_layers)
        elif self.rnn_type == 'rnn':
            self.rnn = RNN(input_size=input_dims, hidden_size=input_dims,
                            num_layers=rnn.num_layers)
        for dim,activation in zip(linear.dims,linear.activations):
            self.mlp.append(Linear(input_dim=input_dims,output_dim=dim))
            if dim!=linear.dims[-1]:
                self.mlp.append(BatchNorm1d(num_feature=dim))
                self.mlp.append(Dropout(p=0.3))
            self.mlp.append(activation_map[activation])
            input_dims = dim
        self.mlp = Sequential(*self.mlp)
    #     self._init_weights()
    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    def forward(self,dvs,depth,state,extras=None):
        depth_f = self.rvt1(depth)
        dvs_f = self.rvt2(dvs)
        out1,out2 = self.crossattention(dvs_f,depth_f)
        feature = out1 + out2
        feature = feature.flatten(start_dim=1)
        hc = None
        if self.has_state:
            x = torch.cat((feature,state),dim=1)
        if self.rnn_type is not None:
            x,hc = self.rnn(x,extras)
        vel = self.mlp(x)
        return vel,hc,None

class FusionCross2(Module):
    def __init__(self,batch_size,image_shape,rvt,has_state,attention,rnn_type,rnn,linear,linear2):
        super().__init__()
        depth_shape = [batch_size,1]+image_shape
        dvs_shape = [batch_size,2]+image_shape
        rvt_config = rvt
        self.rnn_type = rnn_type
        self.has_state = has_state
        self.rvt1 = []
        self.rvt2 = []
        self.mlp = []
        self.mlp2 = []
        for channel,stride in rvt_config:
            rvt1=RVT_block(downsample_channels=channel,pool_kernel_size=stride,pool_stride=stride,use_lstm=False,input_shape=depth_shape)
            self.rvt1.append(rvt1)
            depth_shape = rvt1.output_shape(depth_shape)
            rvt2=RVT_block(downsample_channels=channel,pool_kernel_size=stride,pool_stride=stride,use_lstm=False,input_shape=dvs_shape)
            self.rvt2.append(rvt2)
            dvs_shape = rvt2.output_shape(dvs_shape)
        self.rvt1 = Sequential(*self.rvt1)
        self.rvt2 = Sequential(*self.rvt2)
        _,C,H,W = dvs_shape
        self.crossattention = Bi_CrossAttention(indim1=C,indim2=C,k_dim=attention.k_dim,v_dim=attention.v_dim,num_heads=attention.num_head)
        input_dims = 2*C*H*W
        for dim,activation in zip(linear2.dims,linear2.activations):
            self.mlp2.append(Linear(input_dim=input_dims,output_dim=dim))
            if dim!=linear2.dims[-1]:
                self.mlp2.append(BatchNorm1d(num_feature=dim))
                self.mlp2.append(Dropout(p=0.3))
            self.mlp2.append(activation_map[activation])
            input_dims = dim
        self.mlp2.append(Softmax(dim=-1))
        self.mlp2 = Sequential(*self.mlp2)
        if has_state:
            input_dims = C*H*W+10
        else:
            input_dims = C*H*W
        if self.rnn_type == 'lstm':
            self.rnn = LSTM(input_size=input_dims,hidden_size=input_dims,num_layers=rnn.num_layers)
        elif self.rnn_type == 'gru':
            self.rnn = GRU(input_size=input_dims, hidden_size=input_dims,
                            num_layers=rnn.num_layers)
        elif self.rnn_type == 'rnn':
            self.rnn = RNN(input_size=input_dims, hidden_size=input_dims,
                            num_layers=rnn.num_layers)
        for dim,activation in zip(linear.dims,linear.activations):
            self.mlp.append(Linear(input_dim=input_dims,output_dim=dim))
            if dim!=linear.dims[-1]:
                self.mlp.append(BatchNorm1d(num_feature=dim))
                self.mlp.append(Dropout(p=0.3))
            self.mlp.append(activation_map[activation])
            input_dims = dim
        self.mlp = Sequential(*self.mlp)
    #     self._init_weights()
    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    def forward(self,dvs,depth,state,extras=None):
        depth_f = self.rvt1(depth)
        dvs_f = self.rvt2(dvs)
        out1,out2 = self.crossattention(dvs_f,depth_f)
        con_feature = torch.cat((out1,out2),dim=1)
        con_feature = con_feature.flatten(start_dim=1)
        coff = self.mlp2(con_feature)
        alpha , beta = torch.chunk(coff,2,dim=1)
        feature = alpha*out1.flatten(start_dim=1) + beta*out2.flatten(start_dim=1)
        hc = None
        if self.has_state:
            x = torch.cat((feature,state),dim=1)
        if self.rnn_type is not None:
            x,hc = self.rnn(x,extras)
        vel = self.mlp(x)
        return vel,hc,None