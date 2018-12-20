from torch import nn
import torch

class Flatten(nn.Module):
    def forward(self, input):
        """
        a pytorch version of Flatten layer
        """
        return input.view(input.size(0), -1)

class add_coord(nn.Module):
    def __init__(self):
        super(add_coord,self).__init__()
        
    def forward(self,x):
        bs,ch,h,w = x.size()
        h_coord = torch.range(start = 0,end = h-1).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat([bs,1,1,w])/(h/2)-1
        w_coord = torch.range(start = 0,end = w-1).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat([bs,1,h,1])/(w/2)-1
        x = torch.cat([x,h_coord,w_coord],dim=1)
        return x
    
class Coord2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True):
        """
        Coord Convolution Module
        """
        super(Coord2d,self).__init__()
        self.add_coord = add_coord()
        self.conv = nn.Conv2d(in_channels=in_channels+2,
                              out_channels=out_channels,
                              kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)
        
    def forward(self,x):
        x = self.add_coord(x)
        x = self.conv(x)
        return x
      
class AttLSTM(nn.Module):
    def __init__(self,mask_activation="softmax",**kwargs):
        """
        Attentional LSTM
        input_size: input dimension
        hidden_size: hidden dimension, also the output dimention of LSTM
        other kwargs of LSTM, most of the following is  pilferage from nn.LSTM doc:
            input_size: mentioned above, only have to specify once
            hidden_size: mentioned above, only have to specify once
            num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
                would mean stacking two LSTMs together to form a `stacked LSTM`,
                with the second LSTM taking in outputs of the first LSTM and
                computing the final results. Default: 1
            bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
                Default: ``True``
            batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``False``
            dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
                LSTM layer except the last layer, with dropout probability equal to
                :attr:`dropout`. Default: 0
            bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        """
        super(AttLSTM,self).__init__()
        self.input_size = kwargs["input_size"]
        self.hidden_size = kwargs["hidden_size"]
        self.mask_maker = nn.Linear(self.hidden_size,1)
        self.lstm = nn.LSTM(**kwargs)
        if mask_activation == "softmax":
            self.mask_act = nn.Softmax(dim = 1)
        elif mask_activation == "sigmoid":
            self.mask_act = nn.Sigmoid()
        elif mask_activation == "relu":
            self.mask_act = nn.ReLU()
        elif mask_activation == "passon":
            self.mask_act = passon()
        else:
            print("Activation type:%s not found, should be one of the following:\nsoftmax\nsigmoid\nrelu"%(mask_activation))

    def forward(self,x):
        mask = self.mask_act(self.mask_maker(x).squeeze(-1)).unsqueeze(1) # mask shape (bs,1,seq_leng)
        output, (h_n,c_n) = self.lstm(x)
        output = mask.bmm(output).squeeze(1) # output shape (bs, hidden_size)
        return output, (h_n, c_n), mask.squeeze(1)
    
class passon(nn.Module):
    def __init__(self):
        """
        forward calculation pass on the x
        and do nothing else
        """
        super(passon,self).__init__()
        
    def forward(self,x):
        return x
    
class DeepMaskAttLSTM(nn.Module):
    def __init__(self,mask_activation="softmax",extra_mask_layers=0,**kwargs):
        """
        mask_activation: Activation layer to the mask, right before applying the mask
            one of "softmax","sigmoid","passon", "passon" means no activation
        extra_mask_layers: int number, how many hidden layers does mask_maker use?   
        Attentional LSTM
        input_size: input dimension
        hidden_size: hidden dimension, also the output dimention of LSTM
        other kwargs of LSTM, most of the following is  pilferage from nn.LSTM doc:
            input_size: mentioned above, only have to specify once
            hidden_size: mentioned above, only have to specify once
            num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
                would mean stacking two LSTMs together to form a `stacked LSTM`,
                with the second LSTM taking in outputs of the first LSTM and
                computing the final results. Default: 1
            bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
                Default: ``True``
            batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``False``
            dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
                LSTM layer except the last layer, with dropout probability equal to
                :attr:`dropout`. Default: 0
            bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        """
        super(DeepMaskAttLSTM,self).__init__()
        self.input_size = kwargs["input_size"]
        self.hidden_size = kwargs["hidden_size"]
        self.mask_maker_layers = []
        self.size_in = [self.input_size]+[self.hidden_size]*extra_mask_layers
        for i in range(extra_mask_layers):
            self.mask_maker_layers+=[
                nn.Linear(self.size_in[i],self.hidden_size),
                nn.ReLU(),
                nn.Dropout(),]
        self.mask_maker_layers.append(nn.Linear(self.size_in[-1],1,bias=False))
        self.mask_maker = nn.Sequential(*self.mask_maker_layers)
        self.lstm = nn.LSTM(**kwargs)
        if mask_activation == "softmax":
            self.mask_act = nn.Softmax(dim = 1)
        elif mask_activation == "sigmoid":
            self.mask_act = nn.Sigmoid()
        elif mask_activation == "relu":
            self.mask_act = nn.ReLU()
        elif mask_activation == "passon":
            self.mask_act = passon()
        else:
            print("Activation type:%s not found, should be one of the following:\nsoftmax\nsigmoid\nrelu\npasson"%(mask_activation))

    def forward(self,x):
        mask_basic = self.mask_maker(x).squeeze(-1)
        mask = self.mask_act(mask_basic).unsqueeze(1) # mask shape (bs,1,seq_leng)
        output, (h_n,c_n) = self.lstm(x)
        output = mask.bmm(output)
        output = output.squeeze(1) # output shape (bs, hidden_size)
        return output, (h_n, c_n), mask.squeeze(1)