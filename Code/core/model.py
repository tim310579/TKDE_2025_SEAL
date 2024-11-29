from ast import Global
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torch import Tensor

from core.layers import BaseCNN, BaseRNN, Controller_concat
from core.layers import BaselineNetwork
from core.layers import BaseTCN, Discriminator_GAT, Discriminator_linear
from core.layers import GAT
from core.informer import Informer

from scipy import signal

#, my_transformer #FixedPositionalEncoding, LearnablePositionalEncoding, TransformerBatchNormEncoderLayer, get_pos_encoder, _get_activation_fn #TransformerEncoderLayer, TransformerEncoder

from core.loss import FocalLoss
from torchsummary import summary

import collections
import gc


class SEAL(nn.Module):

    def __init__(self, input_size = 12, hidden_size = 256, hidden_output_size = 1, output_size = 9, 
                  dense1_input=2304, Adj_matrix=None, Corr_matrix=None, embedding_H_t=None, alpha=None, IDL_type=None, backbone=None,
                  nfeat=300, nhid=256, gnn_type=None, cnn_attn=None, window_size=None, device=torch.device('cuda:0')):
                  #feat_dim=12, max_len=500, d_model=64, n_heads=8, num_layers=3, dim_feedforward=256,
                  #dropout=0.1, pos_encoding='fixed', activation='relu', norm='BatchNorm', freeze=False):
    
        super(SEAL, self).__init__()
        
        self.loss_func = FocalLoss(gamma=1, alpha=0.5) #()
        
        self.device = device
        #self.INPUT_SIZE = input_size
        
        self.HIDDEN_SIZE = hidden_size #+ output_size #ViT #+ output_size
        
        
        self.OUTPUT_SIZE = output_size
        
        # for IDL block
        self.IDL_type = 'Informer'
        if IDL_type is not None:
            self.IDL_type = IDL_type
        print(self.IDL_type)
        
        # for CDL block
        self.CELL_TYPE = 'LSTM'
        if backbone is not None:
            self.CELL_TYPE = backbone
        print(self.CELL_TYPE)
        
        # for GNN type
        self.gnn_type = 'GAT'
        if gnn_type is not None:
            self.gnn_type = gnn_type
        print(self.gnn_type)
        
        self.cnn_attn = 'cbam'
        if cnn_attn is not None:
            self.cnn_attn = cnn_attn
        print(cnn_attn)
        
        # tuning parameter --------------------------
        #
        self._ALPHA = alpha
        #
        # tuning parameter --------------------------
        
        self.embedding_H_t = embedding_H_t
        self.Adj_matrix = Adj_matrix
        
        
        self.dropout = 0.5
        
        # ***** IDL block
        if self.IDL_type == 'CNN':
            self.BaseCNN = BaseCNN(input_size, hidden_size, output_size, dense1_input, cnn_attn)#.to(device)#.cuda() # 256
            
        elif self.IDL_type == 'TCNcatCNN':
            self.BaseCNN = BaseCNN(input_size, hidden_size, output_size, dense1_input, cnn_attn)
            self.BaseTCN = BaseTCN(input_size, hidden_size, [24,24,24,24], kernel_size=3, dropout=0.2, only_tcn=True)
            self.HIDDEN_SIZE = hidden_size*2
            
            if input_size == 242: # Opportunity
                self.BaseTCN = BaseTCN(input_size, hidden_size, [256,256,256,256], kernel_size=3, dropout=0.2, only_tcn=True)
            
        elif self.IDL_type == 'Informer':
            self.Informer = Informer(input_size, cnn_attn=cnn_attn, window_size=window_size) # 4->5
            
            if input_size == 242: # Opportunity
                self.Informer = Informer(input_size, e_layers=3, cnn_attn=cnn_attn, window_size=window_size)#, d_model=1024, d_ff=1024)
            
            
        # CDL block
        if self.CELL_TYPE != 'none':
            self.BaseRNN = BaseRNN(self.HIDDEN_SIZE, self.HIDDEN_SIZE, self.CELL_TYPE, num_channels = [8,8,8,8]) #.to(device)#.cuda() # 269 [8,8,16,16], [8, 16, 32, 64]
        
        
        self.Controller_concat = Controller_concat(self.HIDDEN_SIZE, output_size, device=device)#.to(device)#.cuda()
        
        self.BaselineNetwork = BaselineNetwork(self.HIDDEN_SIZE, output_size)#.to(device)#.cuda()
        

        if self.gnn_type == 'GAT':
            self.GAT = GAT(nfeat, nhid, nclass=self.HIDDEN_SIZE, dropout=0.5, alpha=0.2, nheads=8)
            self.Discriminator_GAT = Discriminator_GAT(self.HIDDEN_SIZE, output_size)

        elif self.gnn_type == 'GCN':
            self.GCN = GCN(nfeat, self.HIDDEN_SIZE, dropout=0.5)
            self.Discriminator_GAT = Discriminator_GAT(self.HIDDEN_SIZE, output_size)

        elif self.gnn_type == 'GIN':
            self.GIN = GIN(input_dim=nfeat, hidden_dim=self.HIDDEN_SIZE, output_dim=self.HIDDEN_SIZE, n_layers=2)
            self.Discriminator_GAT = Discriminator_GAT(self.HIDDEN_SIZE, output_size)

        else: # Linear
            self.Discriminator_linear = Discriminator_linear(self.HIDDEN_SIZE, output_size)
        
        
    def initHidden(self, batch_size, weight_size):
        """Initialize hidden states"""
        if self.CELL_TYPE == "LSTM":
            h = (torch.zeros(1, batch_size, weight_size).to(self.device),#.cuda(),
                 torch.zeros(1, batch_size, weight_size).to(self.device))#.cuda())
        else:
            h = torch.zeros(1, batch_size, weight_size).to(self.device)#.cuda()
    
        return h


    def forward(self, X, embedding_H_t=None):

        hidden = self.initHidden(len(X), self.HIDDEN_SIZE)
        hidden_lstmcnn = self.initHidden(len(X), 256)
        
        min_length = 1000
        max_length = 0
        
        for x in X:
            if min_length > x.shape[0]:
                min_length = x.shape[0]
            if max_length < x.shape[0]:
                max_length = x.shape[0]
        
        tau_list = np.zeros(X.shape[0], dtype=int) # record which segment
        tau_list_labels = np.zeros((X.shape[0], self.OUTPUT_SIZE), dtype=int)
        
        state_list = np.zeros((X.shape[0],self.OUTPUT_SIZE), dtype=int)
        
        log_pi = []
        baselines = []
        halt_probs = []
        
        
        for t in range(max_length):
            slice_input = []
            cnn_input = None # cpu
               
            if self.training:        
                for idx, x in enumerate(X):
                    slice_input.append(torch.from_numpy(x[tau_list[idx],:,:]).float())
                    cnn_input = torch.stack(slice_input, dim=0)
            else:
                for idx, x in enumerate(X):
                    slice_input.append(torch.from_numpy(x[tau_list[idx],:,:]).float())
                    cnn_input = torch.stack(slice_input, dim=0)

            cnn_input = cnn_input.to(self.device).detach()
            #print(cnn_input.shape)
            
            
            ############# HAR ###################
            
            
                
            if self.IDL_type == 'Informer':
                S_t = self.Informer(cnn_input)
                
                
            elif self.IDL_type == 'TCNcatCNN':
                S_t1 = self.BaseCNN(cnn_input)
            
                S_t2 = self.BaseTCN(cnn_input)
                
                S_t = torch.cat((S_t1, S_t2), 1)
                
            else: # CNN
                S_t = self.BaseCNN(cnn_input)
            
            #S_t = torch.cat((S_t, y_v), 1)
            
            
            if self.CELL_TYPE != 'none':
                S_t, hidden = self.BaseRNN(S_t.unsqueeze(0), hidden) # Run sequence model
                
                
                if self.CELL_TYPE == "LSTM":
                    S_t = hidden[0][-1]
                else:
                    S_t = hidden[0]
                    
            else:
                #print('here')
                pass
                
            
            if self.gnn_type == 'GAT': ## GAT version
                embedding_H_t = self.GAT(self.embedding_H_t, self.Adj_matrix)
                
                y_hat = self.Discriminator_GAT(S_t, embedding_H_t)
            
            elif self.gnn_type == 'GCN':
                embedding_H_t = self.GCN(self.embedding_H_t, self.Adj_matrix)
                
                y_hat = self.Discriminator_GAT(S_t, embedding_H_t)

            elif self.gnn_type == 'GIN':
                embedding_H_t = self.GIN(self.embedding_H_t, self.Adj_matrix)
                
                y_hat = self.Discriminator_GAT(S_t, embedding_H_t)

            else: # Linear
                y_hat = self.Discriminator_linear(S_t)
            
            rate = t
            if(rate>10): rate = 10
            
            if self.training:
                a_t, p_t, w_t, probs = self.Controller_concat(S_t, y_hat, eps=0, train=True)
            else:
                a_t, p_t, w_t, probs = self.Controller_concat(S_t, y_hat, eps=0, alpha=self._ALPHA)
            
            b_t = self.BaselineNetwork(S_t) # Compute the baseline
            
            baselines.append(b_t)
            
            log_pi.append(p_t)
            
            halt_probs.append(w_t)
            
            for idx, batch_a in enumerate(a_t): #batch
                for id_label, a in enumerate(batch_a):
                    if(a == 0 and tau_list_labels[idx, id_label] < X[idx].shape[0]-1):
                        tau_list_labels[idx, id_label]+=1
                        
                    else:
                        state_list[idx, id_label] = 1
                        
                if tau_list[idx] == X[idx].shape[0]-1: 
                    pass  # segment max limit number
                else:
                    tau_list[idx]+=1
                    
            if (np.mean(state_list)>=1): break # break condition in training phrase
            
        self.log_pi = torch.stack(log_pi).transpose(1, 0).squeeze(2)
        
        self.halt_probs = torch.stack(halt_probs).transpose(1, 0)
        
        self.baselines = torch.stack(baselines).transpose(1, 0)

        self.tau_list = tau_list_labels #tau_list
        
        return y_hat, tau_list_labels
        

    def applyLoss(self, y_hat, labels, alpha = 0.3, beta = 0.001, gamma = 0.5, loss_weight=None, n_js=None):
        # --- compute reward ---
        r = (y_hat.float().round().detach() == labels.float()).float() #checking if it is correct

        r = r*2 - 1 # return 1 if correct and -1 if incorrect
        
        R = torch.from_numpy(np.zeros(self.baselines.shape)).float().to(self.device) #.cuda()
        
        for idx in range(r.shape[0]): # batch
            for id_label in range(r.shape[1]): # labels
                for jdx in range(self.tau_list[idx, id_label]+1): # time point
                    R[idx][jdx][id_label] = r[idx, id_label] * (jdx+1)
        
        # --- subtract baseline from reward ---
        adjusted_reward = R - self.baselines.detach()

        # --- compute losses ---
        self.loss_b = F.mse_loss(self.baselines, R) # Baseline should approximate mean reward
        
        self.loss_c = 0
        
        for i in range(labels.shape[1]): # labels
            tmp_y_hat_1 = torch.unsqueeze(y_hat[:,i], 1)
            tmp_y_hat_0 = torch.from_numpy(np.zeros(tmp_y_hat_1.shape)).float().to(self.device) # .cuda()
            tmp_y_hat_0 = 1 - tmp_y_hat_1
            tmp_y_hat = torch.cat((tmp_y_hat_0, tmp_y_hat_1), 1)

            tmp_y = labels[:,i]
            
            self.loss_c += self.loss_func(tmp_y_hat, tmp_y)#*max_nk/n_js[i] # focal loss
        
        
        self.loss_r = torch.sum(-self.log_pi*adjusted_reward, dim=1).mean() # -1
        self.time_penalty = torch.sum(self.halt_probs, dim=1).mean()
        
        # --- collect all loss terms ---
        loss = self.loss_c + self.loss_r + self.loss_b + beta * self.time_penalty #+ self.loss_v  #+ self.loss_rec 0.01 , 0.001
        
        return loss, self.loss_c, self.loss_r, self.loss_b, self.time_penalty#, self.loss_v #, self.loss_rec
        
