# -*- coding: utf-8 -*-
"""
Code for ST Transformer
"""

import torch
import torch.nn as nn
from GCN_models import GCN
from One_hot_encoder import One_hot_encoder
import torch.nn.functional as F
import numpy as np
import pdb
import pandas as pd
import matplotlib.pyplot as plt

#The dot product attention class
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, count):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 
        '''
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        with open("attentions.csv", "ab") as f:
            f.write(b"\n")
            for p in range(25):
                out_np = np.asarray(attn[0][0][p])
                np.savetxt(f, out_np, delimiter=",")

        context = torch.matmul(attn, V) # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context



class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"



        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V, count):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, T, N, d_k]

        # linear projection of each matrtix, which is the same at first. (all query)
        # input_Q, input_V, and input_K are the same, but Q, K, and V are not because weights of nn.linear
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        #K = self.W_Q(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]
        context = ScaledDotProductAttention()(Q, K, V, count) # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4) #[B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim) # [B, N, T, C]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        return output

class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"


        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V, count):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4) # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        print("Temporal attention...")
        print(count)
        context = ScaledDotProductAttention()(Q, K, V, count) #[B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4) #[B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim) # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        #if it's the last turn, send output.

        return output



class STransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, cheb_K, dropout, forward_expansion):
        super(STransformer, self).__init__()
        # Spatial Embedding
        self.adj = adj
        #self.D_S = adj.to('cuda:0')
        self.D_S = adj
        #self.embed_liner = nn.Linear(adj.shape[0], embed_size)
        self.embed_liner = nn.Linear(64, embed_size)
        self.embed_liner1 = nn.Linear(64, embed_size)
        self.attention = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        # Transfer GCN
        self.gcn = GCN(embed_size, embed_size*2, embed_size, adj, cheb_K, dropout)
        self.norm_adj = nn.InstanceNorm2d(1)    # Normalize the adjacency matrix

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query, count):
        # value, key, query: [N, T, C]  [B, N, T, C]
        # Spatial Embedding part
        # N, T, C = query.shape
        # D_S = self.embed_liner(self.D_S) # [N, C]
        # D_S = D_S.expand(T, N, C) #[T, N, C]相当于在第一维复制了T份
        # D_S = D_S.permute(1, 0, 2) #[N, T, C]


        # Adjacency Matrix Linear Embedding

        # D_S -> torch.Size([25, 64])
        # self.D_S -> torch.Size([25, 25])
        # embed_size = 64
        # Therefore, embed_liner is a single layer feed forward network with size 25 for each input sample and size 64 for each output





        B, N, T, C = query.shape
        # B:50, N:116,  T:25, C:64
        # [B, N, T, C] 50, 64, 25, 64
        # D_S = self.embed_liner(self.D_S) # [N, C] # 64 by 64
        # D_S = D_S.expand(B, T, N, C) #[B, T, N, C]相当于在第2维复制了T份, 第一维复制B份
        # D_S = D_S.permute(0, 2, 1, 3) #[B, N, T, C] # 50, 64, 25, 64 in this case of adj = 64 by 64


        adj_matrix = pd.read_csv('./PEMSD7/correlation1_41_new.csv', header = None)
        #adj_matrix = pd.read_csv('./PEMSD7/adj_TOTAL_Autism.csv', header = None)
        print(adj_matrix.shape)
        k = 0
        adj1 = np.array(adj_matrix[0:64])
        adj1 = self.embed_liner1(torch.Tensor(adj1))
        #adj1 = np.array(adj_matrix[0:116])
        #D_S = torch.zeros(50, 64, 25, 64)
        D_S = torch.zeros(50, 64, 25, 64)

        M=0
        for i in range(50):
            k = k + 64
            adj1 = np.array(adj_matrix[k:k+64])
            adj1 = self.embed_liner1(torch.Tensor(adj1))
            #adj1 = torch.Tensor(adj1)
            adj1 = adj1.expand(T, N, C)
            adj1 = adj1.permute(1, 0, 2)
            #D_S2 = torch.cat((D_S2, D_S2, adj1), 0)
            D_S[M] = adj1

            M = M + 1

        # D_S2 is the new adjaceency matrix: 1198, 64, 25, 64
        # adj1= 64* 64
        # permute= 64 * 25 * 64
        # add to D_S ( batch of 50 )


        #0-64->64-128
        #
        # GCN Part (fixed)
        #self.adj = pd.read_csv('./PEMSD7/Euclidean_adj.csv', header = None)
        self.adj = pd.read_csv('./PEMSD7/Autism_41_Adj.csv', header = None)
        #X_G = torch.Tensor(B, N,  0, C).to('cuda:0')

        self.adj=torch.Tensor(np.array(self.adj))
        X_G = torch.Tensor(B, N,  0, C)

        self.adj = self.adj.unsqueeze(0).unsqueeze(0)
        self.adj = self.norm_adj(self.adj)
        self.adj = self.adj.squeeze(0).squeeze(0)

        #self.adj = self.embed_liner(self.adj)
        # 3200x116 and 64x64- embedded: 116, 64
        for t in range(query.shape[2]):
            # gcn of (query , adjacency)
            #o = self.gcn(query[ : ,:,  t,  : ],  self.adj) # [B, N, C]
            o = self.gcn(query[ : ,:,  t,  : ],  self.adj) # [B, N, C]
            o = o.unsqueeze(2)              # shape [N, 1, C] [B, N, 1, C]
#             print(o.shape)
            #put output of GCN in X_G
            X_G = torch.cat((X_G, o), dim=2)
         # At last X_G [B, N, T, C]


        # Spatial Transformer with attention part

        #Attention of (query + embedded linear adjacency)
        query = query + D_S


        attention = self.attention(query, query, query, count) #(B, N, T, C)

        # Add skip connection, run through normalization and finally dropout

        #feedforward of ( (attention output) +(query + embedded linear adj) )
        x = self.dropout(self.norm1(attention + query))
        # (output of attention -> weighted sum) + query
        #equation 10 : M'S = X^S(query)+ attention
        forward = self.feed_forward(x)
        #dropout of feedforward of (attention + (query + embedded linear adj)) + (attention + (query + embedded linear adj) )
        U_S = self.dropout(self.norm2(forward + x))


        # Fusion STransformer and GCN
        g = torch.sigmoid(self.fs(U_S) +  self.fg(X_G))      # (7)
        out = g*U_S + (1-g)*X_G                                # (8)

        return out #(B, N, T, C)


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()

        self.time_num = time_num
        self.temporal_embedding = nn.Embedding(time_num, embed_size) 



        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, t, count):
        B, N, T, C = query.shape

        D_T = self.temporal_embedding(torch.arange(0, T))
        D_T = D_T.expand(B, N, T, C)


        query = query + D_T
        # concatenate temporal embedding and query
        print ("count")
        print(count)
        attention = self.attention(query, query, query, count)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out




### STBlock

class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, cheb_K, dropout, forward_expansion):
        super(STTransformerBlock, self).__init__()
        self.STransformer = STransformer(embed_size, heads, adj, cheb_K, dropout, forward_expansion)
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, t, count):
    # value,  key, query: [N, T, C] [B, N, T, C]
        # Add skip connection,run through normalization and finally dropout
        #the spatial transformer S extracts spatial features YS from the input node feature XS as well as graph adjacency matrix A.
        #YS = S(XS, A)
        #xs = query?
        #value=key=query
        #query = node features. adj plus node features-> STransformer(value, key, query)
        x1 = self.norm1(self.STransformer(value, key, query, count) + query) #(B, N, T, C)

        x2 = self.dropout( self.norm2(self.TTransformer(x1, x1, x1, t, count) + x1) )
        return x2




### Encoder
class Encoder(nn.Module):
    #  ST-Transformer Block
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        adj,
        time_num,
        device,
        forward_expansion,
        cheb_K,
        dropout,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList(
            [
                STTransformerBlock(
                    embed_size,
                    heads,
                    adj,
                    time_num,
                    cheb_K,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
    # x: [N, T, C]  [B, N, T, C]
        out = self.dropout(x)
        # In the Encoder the query, key, value are all the same.
        count=0
        for layer in self.layers:
            count=count+1
            out = layer(out, out, out, t, count)
            print("layer...")
        #out_np = np.array(out)
        #np.save(out_np, 'attention_out.csv')
        with open("attention_out.csv", "ab") as f:
            f.write(b"\n")
            for p in range(50):
                out_np = np.asarray(out[p][1])
                np.savetxt(f, out_np, delimiter=",")
        #out_np2 = np.asarray(out[2][1])
        #np.savetxt("attention_out.csv", out_np, delimiter=",")
        #np.savetxt("attention_out1.csv", out_np, delimiter=",")
        return out



### Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        adj,
        embed_size,
        num_layers,
        heads,
        time_num,
        forward_expansion, ##？
        cheb_K,
        dropout,
        device = "cpu"
        #device="cuda:0"
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            device,
            forward_expansion,
            cheb_K,
            dropout
        )
        self.device = device

    def forward(self, src, t):
        ## scr: [N, T, C]   [B, N, T, C]
        enc_src = self.encoder(src, t)
        return enc_src # [B, N, T, C]


### ST Transformer: Total Model

class STTransformer(nn.Module):
    def __init__(
        self,
        adj,
        in_channels,
        embed_size,
        time_num,
        num_layers,
        T_dim,
        output_T_dim,
        heads,
        cheb_K,
        forward_expansion,
        dropout = 0
    ):
        super(STTransformer, self).__init__()

        self.forward_expansion = forward_expansion
        # Number of channels for the first convolution expansion
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.Transformer = Transformer(
            adj,
            embed_size,
            num_layers,
            heads,
            time_num,
            forward_expansion,
            cheb_K,
            dropout = 0
        )

        # Reduce the time dimension. Example: T_dim=12 to output_T_dim=3, the input 12 dimensions are reduced to the output 3 dimensions
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        # Reduce the number of channels to 1 dimension
        self.conv3 = nn.Conv2d(embed_size, 1, 1) # 64 to 1 embedding
        self.relu = nn.ReLU()

    def forward(self, x):
        # input x shape[ C, N, T]
        #C: Number of channels. N: The number of sensors. T: amount of time

        # x shape: torch.Size(      [50, 1, 25, 12]) ,   in_channel=1, embed_size=64
        # input_Transformer shape: [50, 64, 25, 12]
        x1 = x[0]
        input_Transformer = self.conv1(x) # conv1= nn.Conv2d(in_channels = 1, embed_size = 64, 1)


        #input_Transformer1 = self.conv1(x1)
        #Conv2d(1, 64, kernel_size=(1, 1), stride=(1, 1))

        input_Transformer = input_Transformer.permute(0, 2, 3, 1)
        #input_Transformer shape becomes ([50, 25, 12, 64])


        #input_Transformer shape[N, T, C]   [B, N, T, C]
        output_Transformer = self.Transformer(input_Transformer, self.forward_expansion)  # [B, N, T, C]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)
        #output_Transformer shape[B, T, N, C]

        # OUTPUT of transformer; Input to prediction
        out = self.relu(self.conv2(output_Transformer))    # Conv2d(T_dim = 12, output_T_dim = 12, 1)  - out shape: [1, output_T_dim, N, C]

        after_relu__matrix=np.array(out[0][0])
        plt.imshow(after_relu__matrix, vmin=0, vmax=1.5, cmap='Greens')
        plt.colorbar()
        plt.show()

        out = out.permute(0, 3, 2, 1)           # out shape: [50, 64, 25, 12]- out shape: [B, C, N, output_T_dim]
        # out shape: [50, 64, 25, 12]


        out = self.conv3(out)                   # Conv2d(embed_size = 64, 1, 1) out shape: [B, 1, N, output_T_dim]
        # out shape: [50, 1, 25, 12])

        out = out.squeeze(1)
        # out shape: [50, 25, 12])

        output_matrix=np.array(out[0])
        plt.imshow(output_matrix, vmin=0, vmax=0.5, cmap='Blues')
        plt.colorbar()
        plt.show()
        return out #[B, N, output_dim]
        # return out shape: [N, output_dim]
