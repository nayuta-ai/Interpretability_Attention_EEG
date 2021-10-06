import numpy as np
import scipy.io as sio
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.graph_attention import GraphAttentionLayer
from preprocess.preprocessed import adjacency_preprocess

model = GraphAttentionLayer(in_features=800*3,out_features=800*3,alpha=0.2,dropout=0.5)
adj = adjacency_preprocess()
data = torch.randn(4,800,32,3)
data = data.permute(0,2,1,3).reshape(4,32,-1)
print(data.shape)
c,e = model(data[0],adj)
c = c.reshape(32,800,3)