import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.distributions import Bernoulli
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


import math

import gc

class BaseCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dense1_input=2304, cnn_attn='cbam'):
        
        super(BaseCNN, self).__init__()
        self.conv= nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5,stride=1)
        self.conv1d= nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2,stride=1)
        
        if input_size == 12:
            self.conv_pad_1_64 =  nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(64,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_2_64 = nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(64,momentum=0.1),
                nn.ReLU()
            )
            
            self.conv_pad_1_128 = nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(128,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_2_128 = nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(128,momentum=0.1),
                nn.ReLU()
            )
            
            self.conv_pad_1_256 = nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(256,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_2_256 = nn.Sequential(
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(256,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_3_256 = nn.Sequential(
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(256,momentum=0.1),
                nn.ReLU()
            )
            
            self.conv_pad_1_512 = nn.Sequential(
                nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(512,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_2_512 = nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(512,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_3_512 = nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(512,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_4_512 = nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(512,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_5_512 = nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(512,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_6_512 = nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(512,momentum=0.1),
                nn.ReLU()
            )
        
        # --------------- Opportunity ------------------
        
        if input_size == 242:
            
            self.conv_pad_HAR_1_256 =  nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(256,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_HAR_2_256 =  nn.Sequential(
                nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(256,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_HAR_3_512 =  nn.Sequential(
                nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(512,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_HAR_4_512 =  nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(512,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_HAR_5_512 =  nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(512,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_HAR_6_512 =  nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(512,momentum=0.1),
                nn.ReLU()
            )
            self.conv_pad_HAR_7_512 =  nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(512,momentum=0.1),
                nn.ReLU()
            )
        
        # --------------- Opportunity ------------------
        
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3,stride=3) 
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3,stride=3) 
        self.maxpool_3 = nn.MaxPool1d(kernel_size=3,stride=3) 
        self.maxpool_4 = nn.MaxPool1d(kernel_size=3,stride=3) 
        self.maxpool_5 = nn.MaxPool1d(kernel_size=3,stride=3)

        
        self.cnn_attn = cnn_attn 
        
        if self.cnn_attn == 'cbam':
            self.cbam = cbam(512, ratio=16)
            
        elif self.cnn_attn == 'SEAttn':
            self.SEAttn = SEAttn(512, reduction_ratio=16)
        
        else:
            pass

        self.dense1 = nn.Linear(dense1_input, 1024)

        self.dense2 = nn.Linear(1024, hidden_size)

    def forward(self, x):
        #------------------------------------------
        #
        # Opportunity
        #
        #------------------------------------------
        
        if x.shape[1] == 30 or x.shape[1] == 60 or x.shape[1] == 40 or x.shape[1] == 50:
            
            x = x.permute(0,2,1)
            
            x = self.conv_pad_HAR_1_256(x)
            x = self.conv_pad_HAR_2_256(x)
            x = self.maxpool_1(x)
    
            x = self.conv_pad_HAR_3_512(x)
            x = self.conv_pad_HAR_4_512(x)
            x = self.maxpool_2(x)
            
            x = self.conv_pad_HAR_5_512(x)
            x = self.conv_pad_HAR_6_512(x)
            x = self.conv_pad_HAR_7_512(x)
            x = self.maxpool_3(x)

            if self.cnn_attn == 'cbam':
                x = self.cbam(x)
                
            elif self.cnn_attn == 'SEAttn':
                x = self.SEAttn(x)
                
            
            h,w = x.shape[1], x.shape[2]
            
            x = x.view(-1, h * w) #Reshape

            x = self.dense1(x)

            x = self.dense2(x)

            return x
        
        #------------------------------------------
        #
        # CPSC_extra, Chapman, PTBXL
        #
        #------------------------------------------
        else:
            x = x.permute(0,2,1)
            
            x = self.conv_pad_1_64(x)
            x = self.conv_pad_2_64(x)
            x = self.maxpool_1(x)
            
            x = self.conv_pad_1_128(x)
            x = self.conv_pad_2_128(x)
            x = self.maxpool_2(x)
            
            x = self.conv_pad_1_256(x)
            x = self.conv_pad_2_256(x)
            x = self.conv_pad_3_256(x)
            x = self.maxpool_3(x)
            
            x = self.conv_pad_1_512(x)
            x = self.conv_pad_2_512(x)
            x = self.conv_pad_3_512(x)
            x = self.maxpool_4(x)

            x = self.conv_pad_4_512(x)
            x = self.conv_pad_5_512(x)
            x = self.conv_pad_6_512(x)
            x = self.maxpool_5(x)

            if self.cnn_attn == 'cbam':
                x = self.cbam(x)
                
            elif self.cnn_attn == 'SEAttn':
                x = self.SEAttn(x)
                
            
            h,w = x.shape[1], x.shape[2]
            
            x = x.view(-1, h * w) #Reshape
            
            x = self.dense1(x)
            
            x = self.dense2(x)
            
            return x


from torch.nn.utils import weight_norm 


class BaseRNN(nn.Module):

    def __init__(self,
                 N_FEATURES,
                 HIDDEN_DIM,
                 CELL_TYPE="LSTM",
                 N_LAYERS=1,
                 num_channels=None): # fo TCN
        super(BaseRNN, self).__init__()
        
        self.CELL_TYPE = CELL_TYPE

        # --- Mappings ---
        
        if CELL_TYPE in ["RNN", "LSTM", "GRU"]:
            
            self.rnn = getattr(nn, CELL_TYPE)(N_FEATURES,
                                              HIDDEN_DIM,
                                              N_LAYERS)
        else:
            if CELL_TYPE == 'TCN':
                self.rnn = BaseTCN(1, HIDDEN_DIM, num_channels, 3, 0.25)
            else:
                try: 
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[CELL_TYPE]
                except KeyError:
                    raise ValueError("""An invalid option for `--model` was
                                     supplied, options are ['LSTM', 'GRU',
                                     'RNN_TANH' or 'RNN_RELU']""")
                
                self.rnn = nn.RNN(N_FEATURES,
                                  HIDDEN_DIM,
                                  N_LAYERS,
                                  nonlinearity=nonlinearity
                                  )
        self.tanh = nn.Tanh()
        #self.LayerNorm = nn.LayerNorm(HIDDEN_DIM)
        

    def forward(self, x_t, hidden):

        if self.CELL_TYPE in ['Transformer']:
            output = self.rnn(x_t, hidden)
            
            return output, output
        
        elif self.CELL_TYPE in ['TCN']:
            output = torch.cat((hidden, x_t), 2)
            
            output = self.rnn(output.permute(1, 2, 0))
            
            output = output.unsqueeze(0)
            
            return output, output
            
        else:
            output, h_t = self.rnn(x_t, hidden)
            
            return output, h_t


class Controller_concat(nn.Module):

    def __init__(self, input_size, output_size, device=None, isCuda = True):
        super(Controller_concat, self).__init__()
        self.isCuda = isCuda
        self.fc = nn.Linear(input_size+output_size, output_size)
        
        
        self.device=device
        
    def forward(self, h_t, y_hat, eps=0.0, train=False, alpha=0.95):
        
        h_t_concat = torch.cat((h_t, y_hat), dim=1)

        probs = torch.sigmoid(self.fc(h_t_concat.detach())) # Compute halting-probability
        
        if(self.isCuda): 
            probs = alpha * probs + (1-alpha) * torch.FloatTensor([eps]).to(self.device)#.cuda() # Add randomness according to eps
                      
        else:
            probs = alpha * probs + (1-alpha) * torch.FloatTensor([eps]) # Add randomness according to eps
        
        probs[probs != probs] = 0
        probs[probs <0] = 0
        probs[probs >1] = 1
        m = Bernoulli(probs=probs) # Define bernoulli distribution parameterized with predicted probability

        ###---------------
        try:
            halt = m.sample() # Sample action
        except:
            halt = 0
        ###---------------
        log_pi = m.log_prob(halt) # Compute log probability for optimization
        
        return halt, log_pi, -torch.log(probs), probs

#####################################################
#############    Discriminators    ##################
#####################################################

class Discriminator_GAT(nn.Module):

    def __init__(self, input_size, output_size):
        super(Discriminator_GAT, self).__init__()
        
        self.fc1 = nn.Linear(input_size*output_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        

    def forward(self, x, h):
        B, L, H = x.shape[0], h.shape[0], x.shape[1] # batch x labels x hidden_size
        
        x = x.unsqueeze(1).expand(B, L, H) #x: B ,H
        h = h.unsqueeze(0).expand(B, L, H) #h : L, H
        
        out = x*h
        
        out = out.view(-1, L*H)
        
        out = self.fc1(out)
        
        y_hat = self.fc2(out)

        return y_hat
       
class Discriminator_linear(nn.Module):

    def __init__(self, input_size, output_size):
        super(Discriminator_linear, self).__init__()
        
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):

        y_hat = self.fc(h_t)
        
        return y_hat
        
#####################################################
##################    GAT    ########################
#####################################################

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        #self.bW = nn.Parameter(torch.empty(size=(32, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        #self.ba = nn.Parameter(torch.empty(size=(32, 2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        #print('adj', adj.device, h.device, self.W.device)
        
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        
        #print('adj', adj.device, e.device, zero_vec.device)
        
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        #print(h.shape, adj.shape, h_prime.shape)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        #print(x.shape)
        #jjj
        return x #F.log_softmax(x, dim=1)

   
#####################################################
##################    GCN    ########################
#####################################################
        
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
    
        support = torch.mm(input, self.weight)
        
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(GCN, self).__init__()
        
        self.gc1 = GraphConvolution(in_features, out_features)
        self.dropout = dropout
        self.gc2 = GraphConvolution(out_features, out_features)
        
    def forward(self, x, adj):
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        
        return x
        
##################################################
######    GIN(Graph Isomorphism Network)    ######
##################################################

class GINConv(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(GINConv, self).__init__()
        
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, A):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix   label x label
        X [batch x nodes x features]: node features matrix  label x hidden
        
        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        
        X = self.linear(X + A @ X)
        X = torch.nn.functional.relu(X)
        
        return X
        
class GIN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super(GIN, self).__init__()
        
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        
        self.convs = torch.nn.ModuleList()
        
        for _ in range(n_layers):
            self.convs.append(GINConv(hidden_dim))
        
        # In order to perform graph classification, each hidden state
        # [batch x nodes x hidden_dim] is concatenated, resulting in
        # [batch x nodes x hiddem_dim*(1+n_layers)], then aggregated
        # along nodes dimension, without keeping that dimension:
        # [batch x hiddem_dim*(1+n_layers)].
        self.out_proj = torch.nn.Linear(hidden_dim*(1+n_layers), output_dim)

    def forward(self, X, A):
        X = self.in_proj(X)

        hidden_states = [X]
        
        for layer in self.convs:
            X = layer(X, A)
            hidden_states.append(X)
            #print(X.shape)

        X = torch.cat(hidden_states, dim=1)#.sum(dim=1)

        X = self.out_proj(X)

        return X
        

def GIN_edge_and_weight(adj_matrix):
    edge_index = []
    edge_weight = []
    num_nodes = adj_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] != 0:
                # �K�[?���_�l??�B?��??�M?��
                edge_index.append([i, j])
                edge_weight.append(adj_matrix[i][j])
    
    edge_index = torch.LongTensor(edge_index).t().contiguous().cuda()
    edge_weight = torch.FloatTensor(edge_weight).cuda()

    return edge_index, edge_weight

##################################################
############    Halting Controller    ############
##################################################

class BaselineNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(input_size, output_size)
        #self.relu = nn.ReLU()

    def forward(self, h_t):
    
        b_t = self.fc(h_t.detach())

        return b_t 



def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        
        
        #self.maxpool = nn.MaxPool1d(kernel_size=3,stride=3) # 3 3 or 2 2

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x) 
        
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res) #self.maxpool(self.relu(out + res))
        
        #out = self.maxpool(out)
        
        #return out
        
        #return self.relu(out)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, # stride=1
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        #self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.network(x)


class BaseTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, only_tcn=True):
        super(BaseTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        
        self.num_channels = num_channels
        self.only_tcn = only_tcn
        
        if self.only_tcn == True:
            self.amp = nn.AdaptiveMaxPool1d(25)
            self.linear1 = nn.Linear(num_channels[-1]*25, 1024)
            self.linear2 = nn.Linear(1024, output_size)
            

    def forward(self, x): # 32 538 1 or 32 500 12
       
        output = self.tcn(x.permute(0, 2, 1))  # 32 16 538 or 32 24 500
        
        if self.only_tcn == True:
            output = self.amp(output) # 32 16 269 or 32 24 256
        
            output = output.view(x.shape[0], -1)
            output = self.linear1(output)
            output = self.linear2(output)
            
            return output
            
        else:
            return output.permute(0, 2, 1)
        
        
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=6):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # 32 12 500 -> 32 12 1
        self.max_pool = nn.AdaptiveMaxPool1d(1) # 32 12 500 -> 32 12 1

        self.sharedMLP = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv1d(in_channels // ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x = x.permute(0,2,1)
        avgout = self.sharedMLP(self.avg_pool(x)) # 32 12 500 -> 32 12 1
        maxout = self.sharedMLP(self.max_pool(x))

        return self.sigmoid(avgout + maxout)
        

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv1d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)  # 32 12 500 -> 32 1 500
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1) # 32 1 500 -> 32 2 500
        x = self.conv(x)
        return self.sigmoid(x)
        
        
class cbam(nn.Module):
    def __init__(self, planes, ratio=6):
        super(cbam, self).__init__()
        self.ca = ChannelAttention(planes, ratio=ratio)
        self.sa = SpatialAttention()
    def forward(self, x):
        
        x = self.ca(x) * x  
        x = self.sa(x) * x
        
        return x
        