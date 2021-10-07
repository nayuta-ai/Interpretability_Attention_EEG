# ref:https://github.com/cedro3/Transformer/blob/master/transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    def __init__(self,d_model=32*4,max_len=3):
        super(PositionalEncoder,self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len,d_model)
        # GPUが使える場合はGPUへ送る
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        
        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False # 勾配を計算しないために

    def forward(self, x):
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret
"""
a = torch.randn(800,3,32*4)
pe = PositionalEncoder()
b = pe(a)
print(b.shape)
"""
class SingleAttention(nn.Module):
    def __init__(self, d_model=32*4):
        super().__init__()
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
        self.d_k = d_model # 大きさを調整する変数

    def forward(self, q, k, v):
        
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        
        # attention
        weights = torch.matmul(q, k.transpose(1,2)) / math.sqrt(self.d_k)
        """
        # mask
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)
        """
        # 規格化
        normalized_weights = F.softmax(weights, dim=-1)
        # attention*v
        output = torch.matmul(normalized_weights, v)
        output = self.out(output)
        return output, normalized_weights

class FeedForward(nn.Module):
    def __init__(self, d_model=32*4, d_ff=1024, dropout=0.1):
        '''Attention層から出力を単純に全結合層2つで特徴量を変換するだけのユニットです'''
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model=32*4, dropout=0.1):
        super().__init__()

        # LayerNormalization層
        # https://pytorch.org/docs/stable/nn.html?highlight=layernorm
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Attention層
        self.attn = SingleAttention(d_model)

        # Attentionのあとの全結合層2つ
        self.ff = FeedForward(d_model)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        # 正規化とAttention
        x_normlized = self.norm_1(x)
        output, normlized_weights = self.attn(
            x_normlized, x_normlized, x_normlized)

        x2 = x + self.dropout_1(output)

        # 正規化と全結合層
        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output, normlized_weights

class TemporalAttentionLayer(nn.Module):
    def __init__(self, pos, transformer):
        super(TemporalAttentionLayer,self).__init__()
        self.pos = pos
        self.transformer = transformer
    def forward(self,x):
        # x: [800, 3, 32, 4]
        x_pos = x.reshape(-1,3,32*4)
        x_pos = self.pos(x_pos)
        y, att = self.transformer(x_pos)
        return y, att
"""
a = torch.randn(800,3,32,4)
pos = PositionalEncoder()
tra = TransformerBlock()
model = TemporalAttentionLayer(pos,tra)
b = model(a)
print(b.shape)
"""