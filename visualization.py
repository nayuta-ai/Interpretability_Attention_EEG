import numpy as np
import mne
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
import torch.nn as nn
from model import SpaceFrequencyTemporalAttention
from module.frequency_attention import FrequencyAttentionLayer
from module.graph_attention import GraphAttentionLayer
from module.temporal_attention import TemporalAttentionLayer,TransformerBlock,PositionalEncoder
from preprocess.preprocessed import adjacency_preprocess
from main import all_preprocess, fix_seed

def topomap(data):
    norm = scipy.stats.zscore(data)
    biosemi_montage = mne.channels.make_standard_montage('biosemi32')
    n_channels = len(biosemi_montage.ch_names)
    fake_info = mne.create_info(ch_names=biosemi_montage.ch_names,sfreq=128.,
                                ch_types='eeg')
    rng = np.random.RandomState(0)
    data_plot = norm[0:32,0:1]
    fake_evoked = mne.EvokedArray(data_plot, fake_info)
    fake_evoked.set_montage(biosemi_montage)
    mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info)
    plt.savefig("./result/topomap.png")
    plt.show()

SEED = 42
fix_seed(SEED)
datasets, labels = all_preprocess("s01","arousal")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datasets = torch.from_numpy(datasets).float().to(device)
adj = adjacency_preprocess(device=device)
FA = FrequencyAttentionLayer(4)
GA = GraphAttentionLayer(in_features=10*3,out_features=10*3,alpha=0.2,dropout=0.5,adj=adj)
tra = TransformerBlock()
pos = PositionalEncoder()
TA = TemporalAttentionLayer(pos,tra)
model = SpaceFrequencyTemporalAttention(device, FA, GA, TA).to(device)
PATH='./param/model.pth'
model.load_state_dict(torch.load(PATH))
out, gr, fr, tr = model(datasets[0:10])
out = out.to('cpu').detach().numpy().copy()
pred_test = np.argmax(out,axis=1)
print(accuracy_score(labels[0:10],pred_test)*100)
print(pred_test)
print(labels[0:10])

# gr visualization
gr = gr.to('cpu').detach().numpy().copy()
freq = np.zeros((4,32))
for i in range(len(gr)):
    for j in range(32):
        freq[i,j] = gr[i,j,j]
topomap(freq[0].reshape(32,1))

# fr visualization
index = ["theta","alpha","beta","gamma"]
fr = fr.to('cpu').detach().numpy().copy()
fr = fr.mean(axis = 0).reshape(4)
plt.figure()
plt.plot(index,fr)
plt.grid()
plt.savefig("./result/frequency_attention.png")
plt.show()

# tr visualization
tr = tr.to('cpu').detach().numpy().copy()
tr = tr.mean(axis=0).reshape(3,3)
t_attention = np.zeros(3)
for i in range(len(tr)):
    t_attention[i] = tr[i,i]
plt.figure()
plt.plot([1,2,3],t_attention)
plt.grid()
plt.savefig("./result/temporal_attention.png")
plt.show()