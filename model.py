import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from module.frequency_attention import FrequencyAttentionLayer
from module.graph_attention import GraphAttentionLayer
from preprocess.preprocessed import adjacency_preprocess

class SpaceFrequencyTemporalAttention(nn.Module):
    def __init__(self,FrequencyAttention,GraphAttention,adjacency):
        super(SpaceFrequencyTemporalAttention,self).__init__()
        self.channel = 4
        self.hidden_dim = 4*3*32
        self.num_labels = 2
        self.FA = FrequencyAttention
        self.GA = GraphAttention
        self.adjacency = adjacency
        self.softmax = nn.Sequential(
            nn.Linear(self.hidden_dim,self.num_labels),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        # x: [800,3,32,4]
        batch = x.shape[0]
        # x = x.permute(3,2,0,1).reshape(4,32,-1)
        """
        y = torch.randn(4,32,batch*3)
        for i in range(self.channel):
            y[i],ga_attention = self.GA(x[i],self.adjacency)
        # y: [4,32,800*3]
        y = y.reshape(4,32,batch,3).permute(2,3,1,0)
        # y: [800,3,32,4]
        """
        z,fr_attention = self.FA(x)
        #z, fr_attention = self.FA(y)
        z = z.reshape(batch,-1)
        s = self.softmax(z)
        return s
"""
a = torch.randn(800,3,32,4)
FA = FrequencyAttentionLayer(4)
GA = GraphAttentionLayer(in_features=800*3,out_features=800*3,alpha=0.2,dropout=0.5)
adj = adjacency_preprocess()
model = SpaceFrequencyTemporalAttention(FA,GA,adj)
b = model(a)
print(b.shape)
"""