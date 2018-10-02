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
    
class attn_lstm(nn.Module):
    def __init__(self,seq_len,vocab_size,hidden_size,num_layers = 1):
        """
        attention layer for sentiment analysis
        seq_len: sequence length
        vocab_size: vocabulary size
        hidden_size: size for embedding, hidden state, cell state
        output: (batch_size, hidden_size)
        """
        super(attn_lstm,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb = nn.Embedding(vocab_size,hidden_size,)
        self.lstm = nn.LSTM(hidden_size,hidden_size,batch_first = True,num_layers = num_layers)
        self.attn = nn.Linear(hidden_size,1,bias = False)
        
    def forward(self,x_input):
        embedded = self.emb(x_input)
        attn_mask = self.attn(embedded)
        output,(h,c) = self.lstm(embedded)
        output = output.permute(0,2,1)
        output_hidden = output.bmm(attn_mask).squeeze(-1)
        return output_hidden, attn_mask
        