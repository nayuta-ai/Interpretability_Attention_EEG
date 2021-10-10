import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from module.frequency_attention import FrequencyAttentionLayer
from module.graph_attention import GraphAttentionLayer
from module.temporal_attention import TemporalAttentionLayer,TransformerBlock,PositionalEncoder
from preprocess.preprocessed import adjacency_preprocess

class SpaceFrequencyTemporalAttention(nn.Module):
    def __init__(self, device, FrequencyAttention, GraphAttention, TemporalAttention):
        super(SpaceFrequencyTemporalAttention,self).__init__()
        self.channel = 4
        self.hidden_dim = 4*3*32
        self.num_labels = 2
        self.FA = FrequencyAttention
        self.GA = GraphAttention
        self.TA = TemporalAttention
        self.softmax = nn.Sequential(
            nn.Linear(self.hidden_dim,self.num_labels),
            nn.Softmax(dim=1)
        )
        self.device = device


    def forward(self,x):
        # x: [800,3,32,4]
        batch = x.shape[0]
        x = x.permute(3,2,0,1).reshape(4,32,-1)
        y = torch.randn(4,32,batch*3).to(self.device)
        gr_attention = torch.randn(4,32,32).to(self.device)
        for i in range(self.channel):
            y[i], gr_attention[i] = self.GA(x[i])
        # y: [4,32,800*3]
        y = y.reshape(4,32,batch,3).permute(2,3,1,0)
        # y: [800,3,32,4]
        #z, fr_attention = self.FA(x)
        z, fr_attention = self.FA(y)
        a, te_attention = self.TA(z)
        z = z.reshape(batch,-1)
        s = self.softmax(z)
        return s, gr_attention, fr_attention, te_attention
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.randn(1,3,32,4).to(device)
adj = adjacency_preprocess(device=device)
FA = FrequencyAttentionLayer(4)
GA = GraphAttentionLayer(in_features=1*3,out_features=1*3,alpha=0.2,dropout=0.5,adj=adj)
tra = TransformerBlock()
pos = PositionalEncoder()
TA = TemporalAttentionLayer(pos,tra)
model = SpaceFrequencyTemporalAttention(device, FA, GA, TA).to(device)
b, g, f, t = model(a)
print(b.shape)
print(g)
print(f)
print(t)
"""